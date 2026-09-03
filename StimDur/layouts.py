from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np

from .config import ViewSpec, StimDurSpec, PlotStyle, StimDurComparisonConfig
from .plots import (
    style_axes, apply_50_tick_labels,
    plot_rt_on_ax, plot_mt_on_ax, plot_psy_on_ax,
)
from .prepare import sem, std
from Juananalysis.kernel_regression import build_hierarchical_data_full, hierarchical_bootstrap_joint
from Juananalysis.plot_results import shaded_curve


def plot_stimdur_4x3_for_view(
    prepared_for_view: Dict[str, dict],          # stimdur_name -> tables
    group_jnd_for_view: Dict[str, "pd.DataFrame"],# stimdur_name -> df
    stimdur_specs: List[StimDurSpec],
    stimdur_colors: Dict[str, str],
    stimdur_pretty: Optional[Dict[str, str]],
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

        if abl ==60:
            ax_rt.set_ylim(0, 0.3)
        else:
            ax_rt.set_ylim(auto=True)   # or just don't set anything
                
        if abl ==20:
            ax_rt.set_ylim(0, 0.7)
        else:
            ax_rt.set_ylim(auto=True)   # or just don't set anything
        


        style_axes(ax_rt, style); style_axes(ax_mt, style); style_axes(ax_psy, style)
        apply_50_tick_labels(ax_rt, cfg.xlim_abs)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

    # legend: stim_dur

    pretty = stimdur_pretty or {}
    legend_labels = [pretty.get(nm, str(nm)) for nm in names]

    handles = [
        plt.Line2D([], [], color=stimdur_colors[nm], marker="o", linestyle="None")
        for nm in names
    ]
    fig.legend(handles, legend_labels, loc="upper right", fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    return fig



# --- One figure per stim_dur, lines = genotypes (views), rows = ABL ---


def plot_genotypes_4x3_for_stimdur(
    prepared: Dict[str, Dict[str, dict]],                 # view_name -> stimdur_name -> tables
    group_jnd: Dict[str, Dict[str, "pd.DataFrame"]],       # view_name -> stimdur_name -> df
    views: List[ViewSpec],
    stimdur_name: str,
    view_colors: Optional[Dict[str, str]],
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
    stimdur_pretty: Optional[Dict[str, str]] = None,       # optional title mapping
    view_pretty: Optional[Dict[str, str]] = None,          # optional legend mapping
) -> plt.Figure:
    """
    Figure for ONE stim_dur:
      - rows = ABL
      - cols = RT / MT / Psychometric
      - lines = views (e.g., genotypes: wt/het/hom)
    Uses the same per-view/per-stimdur prepared tables you already compute.
    """

    if view_colors is None:
        view_colors = {}

    # ---- collect ABL rows across ALL views for this stimdur ----
    abl_set = set()
    for v in views:
        tables = prepared.get(v.name, {}).get(stimdur_name, None)
        if not tables:
            continue
        for key in ("rt_group", "mt_group", "psy_group"):
            dfk = tables.get(key, None)
            if isinstance(dfk, pd.DataFrame) and (not dfk.empty) and ("ABL" in dfk.columns):
                abl_set |= set(dfk["ABL"].unique())

    abl_rows = sorted(int(a) for a in abl_set)
    if not abl_rows:
        raise ValueError(f"No ABL rows found for stimdur='{stimdur_name}' (check filters / stim_dur values).")

    # ---- figure scaffold ----
    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.2 * len(abl_rows)), sharex="col")
    if len(abl_rows) == 1:
        axes = axes.reshape(1, 3)

    stimdur_title = stimdur_name
    if stimdur_pretty is not None:
        stimdur_title = stimdur_pretty.get(stimdur_name, stimdur_name)

    # ---- plot per ABL row ----
    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r, 0], axes[r, 1], axes[r, 2]

        for i, v in enumerate(views):
            tables = prepared.get(v.name, {}).get(stimdur_name, None)
            if not tables:
                continue

            color = view_colors.get(v.name, f"C{i % 10}")

            plot_rt_on_ax(ax_rt, tables, abl=abl, color=color, cfg=cfg)
            plot_mt_on_ax(ax_mt, tables, abl=abl, color=color, cfg=cfg)
            plot_psy_on_ax(ax_psy, tables, abl=abl, color=color, cfg=cfg)

        ax_rt.set_title(f"{stimdur_title} — ABL {abl} RT", fontsize=style.title_fs, pad=style.title_pad)
        ax_mt.set_title(f"{stimdur_title} — ABL {abl} MT", fontsize=style.title_fs, pad=style.title_pad)
        ax_psy.set_title(f"{stimdur_title} — ABL {abl} Psychometric", fontsize=style.title_fs, pad=style.title_pad)

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)

        style_axes(ax_rt, style); style_axes(ax_mt, style); style_axes(ax_psy, style)
        apply_50_tick_labels(ax_rt, cfg.xlim_abs)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

    # ---- legend: genotypes (views) ----
    labels = []
    handles = []
    for i, v in enumerate(views):
        col = view_colors.get(v.name, f"C{i % 10}")
        lab = v.name
        if view_pretty is not None:
            lab = view_pretty.get(v.name, v.name)
        labels.append(lab)
        handles.append(plt.Line2D([], [], color=col, marker="o", linestyle="None"))

    fig.legend(handles, labels, loc="upper right", fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    return fig


def plot_block_conditions_4x3_for_stimdur_and_view(
    prepared_by_condition: Dict[str, Dict[str, dict]],
    stimdur_name: str,
    view_name: str,
    condition_order: Sequence[str],
    condition_colors: Optional[Dict[str, str]],
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
    stimdur_pretty: Optional[Dict[str, str]] = None,
    view_pretty: Optional[Dict[str, str]] = None,
    condition_pretty: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    if condition_colors is None:
        condition_colors = {}

    compact_style = PlotStyle(
        title_fs=max(12, style.title_fs - 7),
        label_fs=max(11, style.label_fs - 6),
        tick_fs=max(10, style.tick_fs - 6),
        legend_fs=max(10, style.legend_fs - 2),
        title_pad=max(8, style.title_pad - 4),
    )

    abl_set = set()
    for condition in condition_order:
        tables = prepared_by_condition.get(condition, {}).get(stimdur_name, None)
        if not tables:
            continue
        for key in ("rt_group", "mt_group", "psy_group"):
            dfk = tables.get(key, None)
            if isinstance(dfk, pd.DataFrame) and (not dfk.empty) and ("ABL" in dfk.columns):
                abl_set |= set(dfk["ABL"].unique())

    abl_rows = sorted(int(a) for a in abl_set)
    if not abl_rows:
        raise ValueError(
            f"No ABL rows found for stimdur='{stimdur_name}', view='{view_name}' in block-condition layout."
        )

    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.2 * len(abl_rows)), sharex="col")
    if len(abl_rows) == 1:
        axes = axes.reshape(1, 3)

    stimdur_title = stimdur_pretty.get(stimdur_name, stimdur_name) if stimdur_pretty else stimdur_name
    view_title = view_pretty.get(view_name, view_name) if view_pretty else view_name

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r, 0], axes[r, 1], axes[r, 2]

        for i, condition in enumerate(condition_order):
            tables = prepared_by_condition.get(condition, {}).get(stimdur_name, None)
            if not tables:
                continue
            color = condition_colors.get(condition, f"C{i % 10}")
            plot_rt_on_ax(ax_rt, tables, abl=abl, color=color, cfg=cfg)
            plot_mt_on_ax(ax_mt, tables, abl=abl, color=color, cfg=cfg)
            plot_psy_on_ax(ax_psy, tables, abl=abl, color=color, cfg=cfg)

        ax_rt.set_title(f"{view_title} | {stimdur_title} | ABL {abl} RT", fontsize=compact_style.title_fs, pad=compact_style.title_pad)
        ax_mt.set_title(f"{view_title} | {stimdur_title} | ABL {abl} MT", fontsize=compact_style.title_fs, pad=compact_style.title_pad)
        ax_psy.set_title(
            f"{view_title} | {stimdur_title} | ABL {abl} Psy",
            fontsize=compact_style.title_fs,
            pad=compact_style.title_pad,
        )

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)

        style_axes(ax_rt, compact_style)
        style_axes(ax_mt, compact_style)
        style_axes(ax_psy, compact_style)
        apply_50_tick_labels(ax_rt, cfg.xlim_abs)
        apply_50_tick_labels(ax_mt, cfg.xlim_sym)
        apply_50_tick_labels(ax_psy, cfg.xlim_sym)

    labels = []
    handles = []
    for i, condition in enumerate(condition_order):
        if condition not in prepared_by_condition:
            continue
        labels.append(condition_pretty.get(condition, condition) if condition_pretty else condition)
        handles.append(
            plt.Line2D([], [], color=condition_colors.get(condition, f"C{i % 10}"), marker="o", linestyle="None")
        )

    fig.legend(handles, labels, loc="upper right", fontsize=compact_style.legend_fs)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    return fig


def plot_biased_metric_grid_for_stimdur(
    prepared_by_condition_and_view: Dict[str, Dict[str, Dict[str, dict]]],
    stimdur_name: str,
    view_names: Sequence[str],
    condition_order: Sequence[str],
    condition_pretty: Optional[Dict[str, str]],
    view_pretty: Optional[Dict[str, str]],
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
    metric: str,
    stimdur_pretty: Optional[Dict[str, str]] = None,
    abl_colors: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    metric = str(metric).lower()
    if metric not in {"rt", "mt", "psy"}:
        raise ValueError("metric must be one of: rt, mt, psy")

    compact_style = PlotStyle(
        title_fs=max(12, style.title_fs - 7),
        label_fs=max(11, style.label_fs - 7),
        tick_fs=max(10, style.tick_fs - 7),
        legend_fs=max(10, style.legend_fs - 2),
        title_pad=max(8, style.title_pad - 4),
    )
    view_names = list(view_names)
    if not view_names:
        raise ValueError("Need at least one view for biased metric grid plotting.")

    abl_values = set()
    for condition in condition_order:
        for view_name in view_names:
            tables = prepared_by_condition_and_view.get(condition, {}).get(view_name, {}).get(stimdur_name)
            if not tables:
                continue
            key = {"rt": "rt_group", "mt": "mt_group", "psy": "psy_group"}[metric]
            dfk = tables.get(key)
            if isinstance(dfk, pd.DataFrame) and not dfk.empty and "ABL" in dfk.columns:
                abl_values |= set(pd.to_numeric(dfk["ABL"], errors="coerce").dropna().astype(int).unique())
    abl_values = sorted(int(a) for a in abl_values)
    if not abl_values:
        raise ValueError(f"No ABL rows found for stimdur='{stimdur_name}' metric='{metric}'.")

    if abl_colors is None:
        abl_colors = {abl: f"C{i % 10}" for i, abl in enumerate(abl_values)}

    condition_colors = {
        "unbiased": "#4D4D4D",
        "rightward": "#1F77B4",
        "leftward": "#D62728",
    }

    n_rows = len(abl_values)
    n_cols = len(view_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.1 * n_rows),
        sharex=False,
        sharey="row",
        squeeze=False,
    )

    metric_label = {"rt": "RT", "mt": "MT", "psy": "Psychometric"}[metric]
    stimdur_title = stimdur_pretty.get(stimdur_name, stimdur_name) if stimdur_pretty else stimdur_name

    def _row_mean_limits(abl: int):
        vals = []
        for view_name in view_names:
            for condition in condition_order:
                tables = prepared_by_condition_and_view.get(condition, {}).get(view_name, {}).get(stimdur_name)
                if not tables:
                    continue
                key = {"rt": "rt_group", "mt": "mt_group", "psy": "psy_group"}[metric]
                dfk = tables.get(key)
                if not isinstance(dfk, pd.DataFrame) or dfk.empty:
                    continue
                sub = dfk[pd.to_numeric(dfk["ABL"], errors="coerce") == int(abl)].copy()
                if sub.empty or "mean" not in sub.columns:
                    continue
                mean_vals = pd.to_numeric(sub["mean"], errors="coerce")
                err_vals = pd.to_numeric(sub["sem"], errors="coerce") if "sem" in sub.columns else pd.Series(0.0, index=sub.index)
                lo = mean_vals - err_vals.fillna(0.0)
                hi = mean_vals + err_vals.fillna(0.0)
                vals.extend(lo[np.isfinite(lo)].tolist())
                vals.extend(hi[np.isfinite(hi)].tolist())
        if not vals:
            return None
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if metric == "psy":
            pad = max(0.03, 0.08 * (hi - lo if hi > lo else 1.0))
            return max(0.0, lo - pad), min(1.0, hi + pad)
        pad = max(0.02, 0.10 * (hi - lo if hi > lo else max(abs(hi), 0.1)))
        if hi <= lo:
            hi = lo + 2 * pad
        return max(0.0, lo - pad), hi + pad

    row_limits = {abl: _row_mean_limits(abl) for abl in abl_values}

    for r, abl in enumerate(abl_values):
        for c, view_name in enumerate(view_names):
            ax = axes[r, c]
            any_tables = False
            for i, condition in enumerate(condition_order):
                tables = prepared_by_condition_and_view.get(condition, {}).get(view_name, {}).get(stimdur_name)
                if not tables:
                    continue
                any_tables = True
                color = condition_colors.get(condition, f"C{i % 10}")
                if metric == "rt":
                    plot_rt_on_ax(ax, tables, abl=abl, color=color, cfg=cfg)
                    ax.set_xlim(*cfg.xlim_abs)
                    apply_50_tick_labels(ax, cfg.xlim_abs)
                elif metric == "mt":
                    plot_mt_on_ax(ax, tables, abl=abl, color=color, cfg=cfg)
                    ax.set_xlim(*cfg.xlim_sym)
                    apply_50_tick_labels(ax, cfg.xlim_sym)
                else:
                    plot_psy_on_ax(ax, tables, abl=abl, color=color, cfg=cfg)
                    ax.set_xlim(*cfg.xlim_sym)
                    apply_50_tick_labels(ax, cfg.xlim_sym)

            if not any_tables:
                ax.set_visible(False)
                continue

            style_axes(ax, compact_style)
            limits = row_limits.get(abl)
            if limits is not None:
                ax.set_ylim(*limits)

            if r == 0:
                ax.set_title(
                    view_pretty.get(view_name, view_name) if view_pretty else view_name,
                    fontsize=compact_style.title_fs,
                    pad=compact_style.title_pad,
                )
            if c == 0:
                ax.set_ylabel(f"ABL {abl}", fontsize=compact_style.label_fs, color="black")

    handles = [
        plt.Line2D([], [], color=condition_colors.get(condition, f"C{i % 10}"), marker="o", linestyle="-")
        for i, condition in enumerate(condition_order)
    ]
    labels = [
        condition_pretty.get(condition, condition) if condition_pretty else condition
        for condition in condition_order
    ]
    fig.legend(handles, labels, loc="upper right", fontsize=compact_style.legend_fs)
    fig.suptitle(
        f"{stimdur_title} {metric_label}",
        fontsize=compact_style.title_fs + 1,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    return fig


def _stimdur_sort_key(name: str) -> tuple[int, float]:
    try:
        value = float(name)
    except Exception:
        return (2, float("inf"))
    if value == 0:
        return (1, float("inf"))
    return (0, value)


def _lighten_color(color: str, frac_to_white: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    frac = float(np.clip(frac_to_white, 0.0, 1.0))
    out = rgb + (1.0 - rgb) * frac
    return tuple(np.clip(out, 0.0, 1.0))


def plot_psychometric_stimdur_grid_for_condition(
    prepared_by_view: Dict[str, Dict[str, dict]],
    stimdur_specs: Sequence[StimDurSpec],
    view_names: Sequence[str],
    condition_name: str,
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
    view_colors: Optional[Dict[str, str]] = None,
    stimdur_pretty: Optional[Dict[str, str]] = None,
    view_pretty: Optional[Dict[str, str]] = None,
    condition_pretty: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    view_colors = view_colors or {}
    stimdur_pretty = stimdur_pretty or {}
    view_pretty = view_pretty or {}
    condition_pretty = condition_pretty or {}
    view_names = list(view_names)

    abl_set = set()
    for view_name in view_names:
        prepared_for_view = prepared_by_view.get(view_name, {})
        for stimdur in stimdur_specs:
            tables = prepared_for_view.get(stimdur.name)
            if not tables:
                continue
            dfk = tables.get("psy_group")
            if isinstance(dfk, pd.DataFrame) and not dfk.empty and "ABL" in dfk.columns:
                abl_set |= set(pd.to_numeric(dfk["ABL"], errors="coerce").dropna().astype(int).unique())
    abl_rows = sorted(int(a) for a in abl_set)
    if not abl_rows:
        raise ValueError(f"No ABL rows found for block condition '{condition_name}'.")

    compact_style = PlotStyle(
        title_fs=max(12, style.title_fs - 6),
        label_fs=max(11, style.label_fs - 6),
        tick_fs=max(10, style.tick_fs - 6),
        legend_fs=max(10, style.legend_fs - 2),
        title_pad=max(8, style.title_pad - 4),
    )
    psy_cfg = StimDurComparisonConfig(
        error_mode="sem",
        skip_psy_fits=cfg.skip_psy_fits,
        xlim_sym=cfg.xlim_sym,
        xlim_abs=cfg.xlim_abs,
        ild_shift_for_abl50=cfg.ild_shift_for_abl50,
    )
    sorted_stimdurs = sorted(list(stimdur_specs), key=lambda spec: _stimdur_sort_key(spec.name))
    n_stim = max(1, len(sorted_stimdurs))

    fig, axes = plt.subplots(
        len(abl_rows),
        len(view_names),
        figsize=(4.8 * max(1, len(view_names)), 4.2 * len(abl_rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for r, abl in enumerate(abl_rows):
        for c, view_name in enumerate(view_names):
            ax = axes[r, c]
            prepared_for_view = prepared_by_view.get(view_name, {})
            any_data = False
            base_color = view_colors.get(view_name, f"C{c % 10}")

            for i, stimdur in enumerate(sorted_stimdurs):
                tables = prepared_for_view.get(stimdur.name)
                if not tables:
                    continue
                psy_group = tables.get("psy_group")
                if not isinstance(psy_group, pd.DataFrame) or psy_group.empty:
                    continue
                sub = psy_group[pd.to_numeric(psy_group["ABL"], errors="coerce") == int(abl)]
                if sub.empty:
                    continue
                any_data = True
                frac = 0.72 * (1.0 - (i / max(1, n_stim - 1))) if n_stim > 1 else 0.0
                color = _lighten_color(base_color, frac)
                plot_psy_on_ax(ax, tables, abl=abl, color=color, cfg=psy_cfg)

            if not any_data:
                ax.set_visible(False)
                continue

            ax.set_xlim(*cfg.xlim_sym)
            style_axes(ax, compact_style)
            apply_50_tick_labels(ax, cfg.xlim_sym)

            if r == 0:
                ax.set_title(
                    view_pretty.get(view_name, view_name),
                    fontsize=compact_style.title_fs,
                    pad=compact_style.title_pad,
                )
            if c == 0:
                ax.set_ylabel(f"ABL {abl}", fontsize=compact_style.label_fs, color="black")

    handles = []
    labels = []
    demo_base = "#4D4D4D"
    for i, stimdur in enumerate(sorted_stimdurs):
        frac = 0.72 * (1.0 - (i / max(1, n_stim - 1))) if n_stim > 1 else 0.0
        handles.append(
            plt.Line2D([], [], color=_lighten_color(demo_base, frac), marker="o", linestyle="-", linewidth=2.5)
        )
        labels.append(stimdur_pretty.get(stimdur.name, stimdur.name))

    fig.legend(handles, labels, loc="upper right", fontsize=compact_style.legend_fs)
    fig.suptitle(
        f"{condition_pretty.get(condition_name, condition_name)} psychometrics by stim duration",
        fontsize=compact_style.title_fs + 1,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.97])
    return fig


def plot_psychometric_stimdur_grid_all_conditions(
    prepared_by_condition_and_view: Dict[str, Dict[str, Dict[str, dict]]],
    stimdur_specs: Sequence[StimDurSpec],
    view_names: Sequence[str],
    condition_order: Sequence[str],
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
    condition_colors: Optional[Dict[str, str]] = None,
    stimdur_pretty: Optional[Dict[str, str]] = None,
    view_pretty: Optional[Dict[str, str]] = None,
    title_suffix: Optional[str] = None,
    shade_reference_specs: Optional[Sequence[StimDurSpec]] = None,
) -> plt.Figure:
    condition_colors = condition_colors or {}
    stimdur_pretty = stimdur_pretty or {}
    view_pretty = view_pretty or {}
    view_names = list(view_names)

    abl_set = set()
    for condition in condition_order:
        for view_name in view_names:
            prepared_for_view = prepared_by_condition_and_view.get(condition, {}).get(view_name, {})
            for stimdur in stimdur_specs:
                tables = prepared_for_view.get(stimdur.name)
                if not tables:
                    continue
                dfk = tables.get("psy_group")
                if isinstance(dfk, pd.DataFrame) and not dfk.empty and "ABL" in dfk.columns:
                    abl_set |= set(pd.to_numeric(dfk["ABL"], errors="coerce").dropna().astype(int).unique())
    abl_rows = sorted(int(a) for a in abl_set)
    if not abl_rows:
        raise ValueError("No ABL rows found for combined psychometric grid.")

    compact_style = PlotStyle(
        title_fs=max(12, style.title_fs - 6),
        label_fs=max(11, style.label_fs - 6),
        tick_fs=max(10, style.tick_fs - 6),
        legend_fs=max(10, style.legend_fs - 2),
        title_pad=max(8, style.title_pad - 4),
    )
    psy_cfg = StimDurComparisonConfig(
        error_mode="sem",
        skip_psy_fits=cfg.skip_psy_fits,
        xlim_sym=cfg.xlim_sym,
        xlim_abs=cfg.xlim_abs,
        ild_shift_for_abl50=cfg.ild_shift_for_abl50,
    )
    sorted_stimdurs = sorted(list(stimdur_specs), key=lambda spec: _stimdur_sort_key(spec.name))
    shade_reference = list(shade_reference_specs) if shade_reference_specs is not None else list(stimdur_specs)
    sorted_reference = sorted(shade_reference, key=lambda spec: _stimdur_sort_key(spec.name))
    reference_names = [spec.name for spec in sorted_reference]
    n_stim = max(1, len(sorted_reference))

    def _shade_frac(stimdur_name: str) -> float:
        if n_stim <= 1:
            return 0.0
        try:
            idx = reference_names.index(stimdur_name)
        except ValueError:
            idx = 0
        return 0.72 * (1.0 - (idx / max(1, n_stim - 1)))

    markers = {"unbiased": "o", "rightward": "s", "leftward": "^"}
    linestyles = {"unbiased": "-", "rightward": "-", "leftward": "-"}

    fig, axes = plt.subplots(
        len(abl_rows),
        len(view_names),
        figsize=(4.8 * max(1, len(view_names)), 4.2 * len(abl_rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for r, abl in enumerate(abl_rows):
        for c, view_name in enumerate(view_names):
            ax = axes[r, c]
            any_data = False

            for condition in condition_order:
                prepared_for_view = prepared_by_condition_and_view.get(condition, {}).get(view_name, {})
                base_color = condition_colors.get(condition, f"C{len(condition)}")
                marker = markers.get(condition, "o")
                linestyle = linestyles.get(condition, "-")

                for i, stimdur in enumerate(sorted_stimdurs):
                    tables = prepared_for_view.get(stimdur.name)
                    if not tables:
                        continue
                    psy_group = tables.get("psy_group")
                    if not isinstance(psy_group, pd.DataFrame) or psy_group.empty:
                        continue
                    sub = psy_group[pd.to_numeric(psy_group["ABL"], errors="coerce") == int(abl)]
                    if sub.empty:
                        continue
                    any_data = True
                    frac = _shade_frac(stimdur.name)
                    color = _lighten_color(base_color, frac)
                    plot_psy_on_ax(
                        ax,
                        tables,
                        abl=abl,
                        color=color,
                        cfg=psy_cfg,
                        marker=marker,
                        linestyle=linestyle,
                        markerfacecolor=color,
                    )

            if not any_data:
                ax.set_visible(False)
                continue

            ax.set_xlim(*cfg.xlim_sym)
            style_axes(ax, compact_style)
            apply_50_tick_labels(ax, cfg.xlim_sym)

            if r == 0:
                ax.set_title(
                    view_pretty.get(view_name, view_name),
                    fontsize=compact_style.title_fs,
                    pad=compact_style.title_pad,
                )
            if c == 0:
                ax.set_ylabel(f"ABL {abl}", fontsize=compact_style.label_fs, color="black")

    condition_handles = [
        plt.Line2D([], [], color=condition_colors.get(condition, "black"), marker=markers.get(condition, "o"), linestyle="-", linewidth=2.5)
        for condition in condition_order
    ]
    condition_labels = [str(condition).capitalize() for condition in condition_order]
    stimdur_handles = []
    stimdur_labels = []
    demo_base = "#4D4D4D"
    for stimdur in sorted_stimdurs:
        frac = _shade_frac(stimdur.name)
        stimdur_handles.append(
            plt.Line2D([], [], color=_lighten_color(demo_base, frac), marker="o", linestyle="-", linewidth=2.5)
        )
        stimdur_labels.append(stimdur_pretty.get(stimdur.name, stimdur.name))

    legend1 = fig.legend(condition_handles, condition_labels, loc="upper right", fontsize=compact_style.legend_fs, bbox_to_anchor=(0.995, 0.995))
    fig.add_artist(legend1)
    fig.legend(stimdur_handles, stimdur_labels, loc="upper right", fontsize=compact_style.legend_fs, bbox_to_anchor=(0.995, 0.80))
    title = "Psychometrics by block condition and stim duration"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    fig.suptitle(title, fontsize=compact_style.title_fs + 1, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.87, 0.97])
    return fig


# --- One row per ABL, one line per ILD, performance over StimDur ---

def _propright_to_pcorrect(ild: float, p_right: float) -> float:
    """Convert PropLeft to P(correct) assuming ILD<0 => Left is correct, ILD>0 => Right is correct."""
    if not np.isfinite(p_right):
        return np.nan
    if ild > 0:
        return p_right
    if ild < 0:
        return 1.0 - p_right
    return 0.5  # ILD==0


def _absild_summary_from_points(points: pd.DataFrame) -> pd.DataFrame:
    """
    points: columns ['subject','ABL','ILD','PropLeft'] (from tables['psy_points']) :contentReference[oaicite:3]{index=3}
    returns df: ['ABL','abs_ILD','mean','sem','n'] where mean/sem are across animals (or std across sessions if 1 animal)
    """
    if points is None or points.empty:
        return pd.DataFrame(columns=["ABL", "abs_ILD", "mean", "sem", "n"])

    df = points.copy()
    df["abs_ILD"] = df["ILD"].abs()
    df["pcorrect"] = [
        _propright_to_pcorrect(ild, pl)
        for ild, pl in zip(df["ILD"].astype(float).values, df["PropLeft"].astype(float).values)
    ]
    df = df[np.isfinite(df["pcorrect"])].copy()
    df = df[df["abs_ILD"] > 0].copy()  # drop ILD==0

    if df.empty:
        return pd.DataFrame(columns=["ABL", "abs_ILD", "mean", "sem", "n"])

    n_subj = df["subject"].nunique() if "subject" in df.columns else 1

    if n_subj <= 1:
        # single-animal mode: treat rows as repeats (sessions), match your style: error=STD
        out = (
            df.groupby(["ABL", "abs_ILD"])["pcorrect"]
            .agg(mean="mean", sem=std, n="count")
            .reset_index()
        )
        return out

    # group mode: first average +/- within each animal (so absILD line uses both signs)
    per_subj = (
        df.groupby(["subject", "ABL", "abs_ILD"])["pcorrect"]
        .mean()
        .reset_index()
        .rename(columns={"pcorrect": "pcorrect_subj"})
    )

    out = (
        per_subj.groupby(["ABL", "abs_ILD"])["pcorrect_subj"]
        .agg(mean="mean", sem=sem, n="count")   # SEM across animals :contentReference[oaicite:4]{index=4}
        .reset_index()
    )
    return out


def _get_or_build_absild_summary(tables: dict) -> pd.DataFrame:
    """
    Reuse cached abs(ILD) summary if available; otherwise build once from psy_points and cache it.
    """
    cached = tables.get("psy_absild_summary", None)
    if isinstance(cached, pd.DataFrame):
        return cached

    points = tables.get("psy_points", None)
    summary = _absild_summary_from_points(points)
    tables["psy_absild_summary"] = summary
    return summary


def plot_absild_perf_across_stimdur_1x3_for_view(
    prepared_for_view: Dict[str, dict],   # stimdur_name -> tables
    stimdur_specs: List[StimDurSpec],
    view_name: str,
    cfg: StimDurComparisonConfig,
    style: PlotStyle,
    stimdur_pretty: Optional[Dict[str, str]] = None,
    abls: Optional[Sequence[int]] = None,        # pick 3 if None
) -> plt.Figure:
    """
    3 panels (ABL). In each panel: one line per abs(ILD),
    x=stimdur, y=P(correct), with errorbars=SEM across animals.
    """
    pretty = stimdur_pretty or {}

    stim_names = [s.name for s in stimdur_specs]
    x = np.arange(len(stim_names))
    xticklabels = [pretty.get(nm, str(nm)) for nm in stim_names]

    # precompute summary per stimdur
    sum_by_sd: Dict[str, pd.DataFrame] = {}
    all_abls = set()
    for nm in stim_names:
        summ = _get_or_build_absild_summary(prepared_for_view[nm])
        sum_by_sd[nm] = summ
        if not summ.empty:
            all_abls |= set(summ["ABL"].astype(int).unique())

    all_abls = sorted(int(a) for a in all_abls if np.isfinite(a))
    all_abls = [a for a in all_abls if a != 50]

    if abls is None:
        if len(all_abls) < 3:
            raise ValueError(f"Need 3 ABLs but found {all_abls}. Pass abls=[...] explicitly.")
        abls = all_abls[:3]
    else:
        abls = [int(a) for a in abls]

    # Wider canvas so each panel gets more x-axis space for tick labels.
    fig, axes = plt.subplots(1, 3, figsize=(39, 8), sharey=True)
    fig.subplots_adjust(
    left=0.06, right=0.99,   # tighter outer margins
    bottom=0.22, top=0.88,   # room for rotated ticks + titles
    wspace=0.12              # keep space between larger panels
)
    for ax, abl in zip(axes, abls):
        # union of abs_ILDs across stimdur for this ABL
        absilds = set()
        for nm in stim_names:
            df = sum_by_sd[nm]
            sub = df[df["ABL"].astype(int) == abl]
            absilds |= set(sub["abs_ILD"].astype(float).unique())
        absilds = sorted(float(v) for v in absilds if np.isfinite(v))

        for absild in absilds:
            y = np.full(len(stim_names), np.nan)
            yerr = np.full(len(stim_names), np.nan)

            for i, nm in enumerate(stim_names):
                df = sum_by_sd[nm]
                row = df[(df["ABL"].astype(int) == abl) & (df["abs_ILD"].astype(float) == absild)]
                if row.empty:
                    continue
                y[i] = float(row["mean"].iloc[0])
                yerr[i] = float(row["sem"].iloc[0])

            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3, label=f"|ILD| {absild:g}")

        ax.set_title(f"{view_name} — ABL {abl}", fontsize=style.title_fs, pad=style.title_pad)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=style.tick_fs)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(correct)", fontsize=style.label_fs)
        style_axes(ax, style)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=1.5)
    return fig


def plot_absild_perf_3x5_all_genotypes(
    prepared: Dict[str, Dict[str, dict]],   # view_name -> stimdur_name -> tables
    views: List[ViewSpec],
    stimdur_specs: List[StimDurSpec],
    abls: Sequence[int],
    style: PlotStyle,
    stimdur_pretty: Optional[Dict[str, str]] = None,
    view_colors: Optional[Dict[str, str]] = None,
    view_pretty: Optional[Dict[str, str]] = None,
    absilds: Optional[Sequence[float]] = None,  # picks first 5 across selected ABLs if None
) -> plt.Figure:
    """
    3x5-style figure (len(abls) rows x 5 columns):
    rows are ABLs, columns are abs(ILD). Each panel has one line per genotype/view.
    """
    pretty = stimdur_pretty or {}
    view_colors = view_colors or {}
    view_pretty = view_pretty or {}
    abls = [int(a) for a in abls]

    stim_names = [s.name for s in stimdur_specs]
    x = np.arange(len(stim_names))
    xticklabels = [pretty.get(nm, str(nm)) for nm in stim_names]

    # Precompute summary tables for each view and stimdur once.
    sum_by_view_sd: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_absilds = set()
    for v in views:
        per_sd: Dict[str, pd.DataFrame] = {}
        for nm in stim_names:
            summ = _get_or_build_absild_summary(prepared[v.name][nm])
            per_sd[nm] = summ
            if not summ.empty:
                sub = summ[summ["ABL"].astype(int).isin(abls)]
                all_absilds |= set(sub["abs_ILD"].astype(float).unique())
        sum_by_view_sd[v.name] = per_sd

    all_absilds = sorted(float(v) for v in all_absilds if np.isfinite(v))
    if absilds is None:
        if len(all_absilds) < 5:
            raise ValueError(
                f"Need at least 5 |ILD| values across ABLs {abls}, found {all_absilds}. "
                "Pass absilds=[...] explicitly."
            )
        absilds = all_absilds[:5]
    else:
        absilds = [float(v) for v in absilds]
        if len(absilds) != 5:
            raise ValueError(f"absilds must contain exactly 5 values, got {len(absilds)}.")

    n_rows = len(abls)
    n_cols = 5
    # Width model in inches:
    # fig width = fixed margins + (n_cols * panel width) + fixed gaps
    # so changing panel_w_in scales only panel width, not inter-panel distance.
    panel_w_in = 5.5
    gap_w_in = 3
    left_in, right_in = 0.6, 1.1
    fig_w_in = left_in + (n_cols * panel_w_in) + ((n_cols - 1) * gap_w_in) + right_in
    fig_h_in = 4.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w_in, fig_h_in), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    wspace = gap_w_in / panel_w_in
    fig.subplots_adjust(
        left=left_in / fig_w_in,
        right=1.0 - (right_in / fig_w_in),
        bottom=0.06,
        top=0.88,
        wspace=wspace,
        hspace=0.35,
    )

    for r, abl in enumerate(abls):
        for c, absild in enumerate(absilds):
            ax = axes[r, c]
            for i, v in enumerate(views):
                y = np.full(len(stim_names), np.nan)
                yerr = np.full(len(stim_names), np.nan)

                for j, nm in enumerate(stim_names):
                    df = sum_by_view_sd[v.name][nm]
                    row = df[
                        (df["ABL"].astype(int) == int(abl))
                        & (df["abs_ILD"].astype(float) == float(absild))
                    ]
                    if row.empty:
                        continue
                    y[j] = float(row["mean"].iloc[0])
                    yerr[j] = float(row["sem"].iloc[0])

                color = view_colors.get(v.name, f"C{i % 10}")
                label = view_pretty.get(v.name, v.name)
                ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3, color=color, label=label)

            ax.set_title(f"ABL {abl} | |ILD| {absild:g}", fontsize=style.title_fs, pad=style.title_pad)
            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=style.tick_fs)
            else:
                ax.tick_params(labelbottom=False)
            ax.set_ylim(0, 1)
            if c == 0:
                ax.set_ylabel("P(correct)", fontsize=style.label_fs)
            style_axes(ax, style, square=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=style.legend_fs)
    fig.suptitle("Performance by |ILD| across StimDur (all ABLs)", fontsize=style.title_fs + 2, y=0.98)
    return fig






# --- One figure per genotype, row per ABL, one line per ILD, performance over StimDur ---




def plot_kreg_4x3_by_abl_for_view(
    df_view: pd.DataFrame,
    view_name: str,
    stimdur_specs,
    stimdur_col: str,
    stimdur_colors,
    stimdur_pretty=None,
    *,
    abls=(20, 40, 60),
    xxi=None,
    h=0.015,
    B=1000,
    xlim=(0.0, 0.5),
    debug: bool = False,  # <-- added so your call can pass debug=True
):
    # If you didn’t pass xxi explicitly, make it match the xlim window by default.
    if xxi is None:
        if xlim is None:
            xxi = np.linspace(0.0, 1.0, 200)
        else:
            xxi = np.linspace(float(xlim[0]), float(xlim[1]), 200)

    df = df_view.copy()

    # ---------- column mapping to what kernel_regression expects ----------
    if "RT" not in df.columns and "timed_rt" in df.columns:
        df["RT"] = df["timed_rt"]
    if "MT" not in df.columns and "timed_mt" in df.columns:
        df["MT"] = df["timed_mt"]
    if "ILD" not in df.columns and "stim_ild" in df.columns:
        df["ILD"] = df["stim_ild"]
    if "ABL" not in df.columns and "stim_abl" in df.columns:
        df["ABL"] = df["stim_abl"]

    # success -> Out (robust to -1/1 or 0/1)
    if "Out" not in df.columns:
        if "success" not in df.columns:
            raise KeyError("Need 'success' or 'Out' in df_view for kreg.")
        s = pd.to_numeric(df["success"], errors="coerce")
        df["Out"] = np.where(s == 1, 1.0, np.where((s == -1) | (s == 0), 0.0, np.nan))

    required = ["animal", "session", "RT", "MT", "ILD", "ABL", "Out", stimdur_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for kreg: {missing}\nHave: {list(df.columns)}")

    # numeric conversions
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["MT"] = pd.to_numeric(df["MT"], errors="coerce")
    df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
    df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
    df[stimdur_col] = pd.to_numeric(df[stimdur_col], errors="coerce")

    df = df.dropna(subset=required).copy()

    # ---------- unit guard: if RT looks like ms, convert RT/MT to seconds ----------
    rt_med = float(np.nanmedian(df["RT"].to_numpy()))
    if rt_med > 5:
        if debug:
            print(f"[{view_name}] RT looks like ms (median={rt_med:.2f}); converting RT/MT to seconds.")
        df["RT"] = df["RT"] / 1000.0
        df["MT"] = df["MT"] / 1000.0

    # constant group so build_hierarchical_data_full cannot filter away anything
    df["_KREG_GROUP_"] = 1

    fig, axs = plt.subplots(
        4, len(abls),
        figsize=(4.2 * len(abls), 10),
        sharex="col",
        sharey="row",
    )

    def _any_finite(y: np.ndarray) -> bool:
        y = np.asarray(y, dtype=float)
        if xlim is None:
            return np.isfinite(y).any()
        m = (xxi >= float(xlim[0])) & (xxi <= float(xlim[1]))
        return np.isfinite(y[m]).any()

    any_plotted = False

    for c, abl in enumerate(abls):
        abl_val = float(abl)

        # ABL selection
        df_a = df[np.isclose(df["ABL"].to_numpy(), abl_val)].copy()
        if debug:
            print(f"[{view_name}] ABL {abl}: rows={len(df_a)}")

        for s in stimdur_specs:
            # ✅ FIX: use the StimDurSpec selector (exact same selection as the rest of StimDur)
            if hasattr(s, "selector") and callable(getattr(s, "selector")):
                df_as = s.selector(df_a).copy()
            else:
                # fallback if selector doesn't exist
                sd_val = getattr(s, "stim_dur", None)
                if sd_val is None:
                    try:
                        sd_val = float(s.name)
                    except Exception:
                        sd_val = s.name
                if isinstance(sd_val, (int, float, np.integer, np.floating)):
                    df_as = df_a[np.isclose(df_a[stimdur_col].to_numpy(), float(sd_val))].copy()
                else:
                    df_as = df_a[df_a[stimdur_col].astype(str) == str(sd_val)].copy()

            if debug:
                print(f"   SD {s.name}: rows={len(df_as)}")

            if df_as.empty:
                continue

            nested = build_hierarchical_data_full(
                df_as,
                group_col="_KREG_GROUP_",
                group_value=1,
                easy_value=None,
                abl_value=None,  # already filtered by df_a
            )

            RTD, TCM, CDF, MTcur = hierarchical_bootstrap_joint(nested, xxi, h, B)

            color = stimdur_colors.get(s.name, None)
            label = stimdur_pretty.get(s.name, s.name) if stimdur_pretty else s.name

            # Only mark as plotted if there are finite values to actually draw
            if _any_finite(TCM[0]) or _any_finite(MTcur[0]) or _any_finite(RTD[0]) or _any_finite(CDF[0]):
                any_plotted = True

            plt.sca(axs[0, c]); shaded_curve(xxi, *TCM,   color=color, label=label)
            plt.sca(axs[1, c]); shaded_curve(xxi, *MTcur, color=color, label=label)
            plt.sca(axs[2, c]); shaded_curve(xxi, *RTD,   color=color, label=label)
            plt.sca(axs[3, c]); shaded_curve(xxi, *CDF,   color=color, label=label)

        axs[0, c].set_title(f"ABL {abl}")

    # cosmetics
    for c in range(len(abls)):
        for r in range(4):
            if xlim is not None:
                axs[r, c].set_xlim(*xlim)
            axs[r, c].grid(axis="x", linestyle=":", alpha=0.5)

    axs[0, 0].set_ylabel("P(correct)")
    axs[1, 0].set_ylabel("MT (s)")
    axs[2, 0].set_ylabel("density")
    axs[3, 0].set_ylabel("CDF")
    for c in range(len(abls)):
        axs[3, c].set_xlabel("RT (s)")

    axs[0, 0].set_ylim(0, 1)
    axs[3, 0].set_ylim(0, 1)

    axs[1, 0].set_ylim(0.15, 0.5)
    axs[1, 1].set_ylim(0.15, 0.5)
    axs[1, 2].set_ylim(0.15, 0.5)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title=stimdur_col)

    fig.suptitle(f"{view_name}")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    if not any_plotted:
        raise RuntimeError(
            f"Selection produced data (see debug), but curves contain no finite values within xlim={xlim}. "
            f"Try widening xlim (e.g., xlim=(0,1.5)) or passing xxi that spans your RT range."
        )

    return fig
