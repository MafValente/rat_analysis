# groupcomparison_modular.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import Psychometric
import Helpers.DataHelpers as DataHelpers  # your file


# ----------------------------
# Config / Specs
# ----------------------------

Selector = Callable[[pd.DataFrame], pd.DataFrame]

@dataclass(frozen=True)
class ViewSpec:
    name: str
    selector: Selector

@dataclass(frozen=True)
class FilterConfig:
    training_min: int = 16
    session_min: int = 13
    drop_repeat_trials: bool = True

@dataclass(frozen=True)
class PlotStyle:
    title_fs: int = 24
    label_fs: int = 25
    tick_fs: int = 24
    legend_fs: int = 16
    title_pad: int = 16

@dataclass(frozen=True)
class GroupComparisonConfig:
    error_mode: str = "individuals"   # "sem" or "individuals"
    skip_psy_fits: Tuple[int, ...] = (50,)
    xlim_sym: Tuple[float, float] = (-19, 19)  # for signed axes
    xlim_abs: Tuple[float, float] = (0, 19)    # for abs axes like RT
    ild_shift_for_abl50: bool = True

@dataclass(frozen=True)
class OverlaySpec:
    """Optional overlays from old neurotypical pickles."""
    makefig1_data: Optional[dict] = None        # fig1_plot_data.pkl
    makefig1_chrono: Optional[dict] = None      # fig1_chrono_plot_data.pkl
    overlay_color: str = "black"


# ----------------------------
# Small utilities
# ----------------------------

def sem(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else x.std(ddof=1) / np.sqrt(len(x))

def _maybe_shift(x, cfg: GroupComparisonConfig):
    if not cfg.ild_shift_for_abl50:
        return np.asarray(x, dtype=float)
    return DataHelpers.shift_ILD_for_ABL50(x)

def apply_filters(df: pd.DataFrame, fcfg: FilterConfig) -> pd.DataFrame:
    df = df.copy()
    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
    if fcfg.drop_repeat_trials and "trial_is_repeat" in df.columns:
        df = df[df["trial_is_repeat"] == False].copy()
    if "training_level" in df.columns:
        df = df[df["training_level"] >= fcfg.training_min]
    if "session" in df.columns:
        df = df[df["session"] >= fcfg.session_min]
    return df


# ----------------------------
# Preparation functions (reusable)
# ----------------------------

def prep_rt(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_s = df_in[df_in["success"] == 1].copy()
    df_s["abs_ILD"] = df_s["ILD"].abs()

    per_subj = (
        df_s.groupby(["animal", "ABL", "abs_ILD"])["timed_rt"]
        .agg(mean_rt="mean")
        .reset_index()
        .rename(columns={"abs_ILD": "ILD"})
    )

    grouped = (
        per_subj.groupby(["ABL", "ILD"])
        .agg(mean=("mean_rt", "mean"), sem=("mean_rt", sem), n=("mean_rt", "count"))
        .reset_index()
    )
    return per_subj, grouped


def prep_mt(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_subj = (
        df_in[df_in["success"] == 1]
        .groupby(["animal", "ABL", "ILD"])["timed_mt"]
        .agg(mean_mt="mean")
        .reset_index()
    )

    grouped = (
        per_subj.groupby(["ABL", "ILD"])
        .agg(mean=("mean_mt", "mean"), sem=("mean_mt", sem), n=("mean_mt", "count"))
        .reset_index()
    )
    return per_subj, grouped


def prep_psy(df_in: pd.DataFrame, do_individual_fits: bool) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    all_pts = []
    per_subject_curves = {}  # (subject, abl) -> dict(xx, yy)

    for subject, df_subj in df_in.groupby("animal"):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
        for abl, res in results.items():
            ILDs = np.asarray(res["ILDs"])
            pleft = np.asarray(res["PropLeft"], dtype=float)

            for ild, val in zip(ILDs, pleft):
                all_pts.append({"subject": subject, "ABL": abl, "ILD": ild, "PropLeft": val})

            if do_individual_fits and len(ILDs) >= 4 and np.isfinite(pleft).all():
                n_trials = np.full_like(ILDs, 50)
                try:
                    _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="my_psycho", n_trials=n_trials, show_plot=False
                    )
                    per_subject_curves[(subject, abl)] = dict(xx=xx, yy=yy)
                except Exception:
                    pass

    points = pd.DataFrame(all_pts)

    agg = (
        points.groupby(["ABL", "ILD"])
        .agg(mean=("PropLeft", "mean"), sem=("PropLeft", sem), n=("PropLeft", "count"))
        .reset_index()
    )

    mean_fits = {}
    for abl in sorted(agg["ABL"].unique()):
        sub = agg[agg["ABL"] == abl]
        ILDs, y = sub["ILD"].values, sub["mean"].values
        n_trials = np.full_like(ILDs, 50)
        try:
            _, _, xx, yy = Psychometric.fit_and_plot_psychometric(
                ILDs, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
            mean_fits[abl] = dict(xx=xx, yy=yy)
        except Exception:
            mean_fits[abl] = None

    return points, agg, per_subject_curves, mean_fits


def build_prepared(df: pd.DataFrame, views: List[ViewSpec], cfg: GroupComparisonConfig) -> Dict[str, dict]:
    prepared = {}
    for v in views:
        df_v = v.selector(df.copy())
        rt_per_subj, rt_group = prep_rt(df_v)
        mt_per_subj, mt_group = prep_mt(df_v)
        psy_points, psy_group, psy_indiv, psy_mean = prep_psy(
            df_v, do_individual_fits=(cfg.error_mode == "individuals")
        )
        prepared[v.name] = dict(
            rt_per_subj=rt_per_subj, rt_group=rt_group,
            mt_per_subj=mt_per_subj, mt_group=mt_group,
            psy_points=psy_points, psy_group=psy_group,
            psy_indiv_curves=psy_indiv, psy_mean_fits=psy_mean,
            df_view=df_v,  # keep for JND extraction
        )
    return prepared


def compute_group_jnd(prepared: Dict[str, dict], skip_abl: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Returns {view_name: group_jnd_df} where df has columns: ABL, mean, sem, n
    """
    out = {}
    for view_name, tables in prepared.items():
        df_v = tables["df_view"]
        all_jnds = []
        for subject, df_subj in df_v.groupby("animal"):
            results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
            jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_abl)
            if jnd_df is None or jnd_df.empty:
                continue
            jnd_df = jnd_df.copy()
            jnd_df["subject"] = subject
            all_jnds.append(jnd_df)

        if not all_jnds:
            out[view_name] = pd.DataFrame(columns=["ABL", "mean", "sem", "n"])
            continue

        all_jnds_df = pd.concat(all_jnds, ignore_index=True)
        group_jnd = (
            all_jnds_df.groupby("ABL")["JND"]
            .agg(mean="mean", sem=sem, n="count")
            .reset_index()
        )
        out[view_name] = group_jnd
    return out


# ----------------------------
# Plot primitives (reusable)
# ----------------------------

def _style_axes(ax, style: PlotStyle, title=None, xlabel=None, ylabel=None, square=True):
    if title:
        ax.set_title(title, fontsize=style.title_fs, pad=style.title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style.label_fs, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style.label_fs, color="black")

    ax.tick_params(axis="both", which="major", labelsize=style.tick_fs,
                   colors="black", width=1.5, length=6)

    for s in ["left", "right", "top", "bottom"]:
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if square and hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)


def _apply_50_tick_labels(ax):
    # ensure ±18 are present and label them ±50
    xticks = sorted(set(ax.get_xticks()) | {-18, 18})
    ax.set_xticks(xticks)
    ax.set_xticklabels([("-50" if x == -18 else "50" if x == 18 else str(int(x))) for x in xticks])


def plot_rt_on_ax(ax, tables: dict, abl: int, color: str, cfg: GroupComparisonConfig):
    rt_group = tables["rt_group"]
    rt_per_subj = tables["rt_per_subj"]

    if cfg.error_mode == "sem":
        sub = rt_group[rt_group["ABL"] == abl]
        if sub.empty:
            return
        x = _maybe_shift(sub["ILD"], cfg)
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                    markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
        return

    # individuals: faint per subject, then mean ± SEM + line
    sub_ps = rt_per_subj[rt_per_subj["ABL"] == abl]
    for _, df_an in sub_ps.groupby("animal"):
        df_an = df_an.sort_values("ILD")
        ax.plot(_maybe_shift(df_an["ILD"], cfg), df_an["mean_rt"],
                color=color, alpha=0.35, linewidth=1.5)

    sub = rt_group[rt_group["ABL"] == abl].sort_values("ILD")
    if sub.empty:
        return
    x = _maybe_shift(sub["ILD"], cfg)
    ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    ax.plot(x, sub["mean"], color=color, linewidth=2.5)


def plot_mt_on_ax(ax, tables: dict, abl: int, color: str, cfg: GroupComparisonConfig):
    mt_group = tables["mt_group"]
    mt_per_subj = tables["mt_per_subj"]

    if cfg.error_mode == "sem":
        sub = mt_group[mt_group["ABL"] == abl]
        if sub.empty:
            return
        x = _maybe_shift(sub["ILD"], cfg)
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                    markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
        return

    sub_ps = mt_per_subj[mt_per_subj["ABL"] == abl]
    for _, df_an in sub_ps.groupby("animal"):
        df_an = df_an.sort_values("ILD")
        ax.plot(_maybe_shift(df_an["ILD"], cfg), df_an["mean_mt"],
                color=color, alpha=0.35, linewidth=1.5)

    sub = mt_group[mt_group["ABL"] == abl].sort_values("ILD")
    if sub.empty:
        return
    x = _maybe_shift(sub["ILD"], cfg)
    ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    ax.plot(x, sub["mean"], color=color, linewidth=2.5)


def plot_psy_on_ax(ax, tables: dict, abl: int, color: str, cfg: GroupComparisonConfig):
    psy_group = tables["psy_group"]
    psy_indiv = tables["psy_indiv_curves"]
    psy_mean  = tables["psy_mean_fits"]

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1)

    sub = psy_group[psy_group["ABL"] == abl]
    if not sub.empty:
        x_vals = _maybe_shift(sub["ILD"], cfg)
        ax.errorbar(x_vals, sub["mean"], yerr=sub["sem"],
                    fmt="o", color=color, markersize=8.5,
                    linewidth=0, elinewidth=1.5, capsize=3)

    if abl in set(cfg.skip_psy_fits):
        return

    if cfg.error_mode == "individuals":
        for (subject, abl_key), curve in psy_indiv.items():
            if abl_key != abl:
                continue
            ax.plot(_maybe_shift(curve["xx"], cfg), curve["yy"],
                    color=color, alpha=0.3, linewidth=1)

    mean_fit = psy_mean.get(abl)
    if mean_fit:
        ax.plot(_maybe_shift(mean_fit["xx"], cfg), mean_fit["yy"],
                color=color, linewidth=(3 if cfg.error_mode == "individuals" else 2))


# ----------------------------
# Layouts
# ----------------------------

def plot_views_3x3(
    prepared: Dict[str, dict],
    views: List[ViewSpec],
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    overlay: OverlaySpec,
) -> plt.Figure:
    """
    Rows = views (genotypes), Cols = RT/MT/Psy
    Each axis shows multiple ABLs in ABL colors.
    """
    view_names = [v.name for v in views]

    # ABLs across all views
    abls = sorted(set().union(*[set(prepared[vn]["rt_group"]["ABL"].unique()) for vn in view_names]))
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
                    color=overlay.overlay_color, show_individuals=False, use_abl_colors=False
                )

        _style_axes(ax_rt, style, f"{vn} — RT", "ILD (dB)", "Mean RT (s)")
        _style_axes(ax_mt, style, f"{vn} — MT", "ILD (dB)", "Mean MT (s)")
        _style_axes(ax_psy, style, f"{vn} — Psychometric", "ILD (dB)", "P(Left)")

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)
        _apply_50_tick_labels(ax_psy)
        _apply_50_tick_labels(ax_mt)

    # global legend = ABL colors
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
    view_colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Rows = ABLs, Cols = RT/MT/Psy
    Each row shows multiple views (genotypes) in view colors.
    """
    view_names = [v.name for v in views]

    # ABL rows across all views
    abl_rows = sorted(set().union(*[set(prepared[vn]["rt_group"]["ABL"].unique()) for vn in view_names]))

    if view_colors is None:
        base = ["C0", "C1", "C2", "C3", "C4"]
        view_colors = {vn: base[i % len(base)] for i, vn in enumerate(view_names)}

    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.8 * len(abl_rows)), squeeze=False)

    for r, abl in enumerate(abl_rows):
        ax_rt, ax_mt, ax_psy = axes[r]

        for vn in view_names:
            c = view_colors.get(vn, "gray")
            tables = prepared[vn]
            plot_rt_on_ax(ax_rt, tables, abl, c, cfg)
            plot_mt_on_ax(ax_mt, tables, abl, c, cfg)
            plot_psy_on_ax(ax_psy, tables, abl, c, cfg)

        # overlays
        if overlay.makefig1_chrono is not None:
            DataHelpers.overlay_makefig1_rt(ax_rt, abl, overlay.makefig1_chrono, color=overlay.overlay_color, zorder=-1)
        if overlay.makefig1_data is not None and abl != 50:
            DataHelpers.overlay_makefig1_psychometrics(
                ax_psy, overlay.makefig1_data, abl=abl,
                color=overlay.overlay_color, show_individuals=False, use_abl_colors=False
            )

        _style_axes(ax_rt, style, f"ABL {abl} — RT", "ILD (dB)", "Mean RT (s)")
        _style_axes(ax_mt, style, f"ABL {abl} — MT", "ILD (dB)", "Mean MT (s)")
        _style_axes(ax_psy, style, f"ABL {abl} — Psychometric", "ILD (dB)", "P(Left)")

        ax_rt.set_xlim(*cfg.xlim_abs)
        ax_mt.set_xlim(*cfg.xlim_sym)
        ax_psy.set_xlim(*cfg.xlim_sym)
        _apply_50_tick_labels(ax_psy)
        _apply_50_tick_labels(ax_mt)

    # legend = views
    handles = [plt.Line2D([], [], color=view_colors[vn], marker="o", linestyle="None") for vn in view_names]
    fig.legend(handles, view_names, loc="upper center", ncol=min(6, len(view_names)), fontsize=style.legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_jnd_figure(
    group_jnd_by_view: Dict[str, pd.DataFrame],
    views: List[ViewSpec],
    style: PlotStyle,
    view_colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Separate JND figure: one panel per view (1xN).
    """
    view_names = [v.name for v in views]
    if view_colors is None:
        base = ["C0", "C1", "C2", "C3", "C4"]
        view_colors = {vn: base[i % len(base)] for i, vn in enumerate(view_names)}

    fig, axes = plt.subplots(1, len(view_names), figsize=(4.2 * len(view_names), 4.0), squeeze=False)
    for i, vn in enumerate(view_names):
        ax = axes[0, i]
        dfj = group_jnd_by_view.get(vn, pd.DataFrame())
        if dfj is None or dfj.empty:
            _style_axes(ax, style, title=f"{vn} — JND", xlabel="ABL", ylabel="JND (dB)")
            ax.set_box_aspect(1)
            continue

        c = view_colors.get(vn, "gray")
        for _, row in dfj.iterrows():
            ax.errorbar(
                row["ABL"], row["mean"], yerr=row["sem"],
                fmt="o", color=c, markersize=7,
                elinewidth=1.5, capsize=4,
                markeredgecolor="black", markeredgewidth=1,
            )

        _style_axes(ax, style, title=f"{vn} — JND", xlabel="ABL", ylabel="JND (dB)")
        ax.set_xticks(sorted(dfj["ABL"].unique()))
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ----------------------------
# Runner (single entry point)
# ----------------------------

def run_groupcomparison(
    cohort_csv: str,
    views: List[ViewSpec],
    cfg: GroupComparisonConfig = GroupComparisonConfig(),
    fcfg: FilterConfig = FilterConfig(),
    style: PlotStyle = PlotStyle(),
    overlay: OverlaySpec = OverlaySpec(),
    layout: str = "views_3x3",   # "views_3x3" or "abls_4x3"
    view_colors: Optional[Dict[str, str]] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Returns dict with figures + prepared tables so you can reuse downstream.
    """
    df = pd.read_csv(cohort_csv)
    df = apply_filters(df, fcfg)

    prepared = build_prepared(df, views, cfg)
    group_jnd_by_view = compute_group_jnd(prepared, skip_abl=50)

    # figures
    if layout == "views_3x3":
        fig_main = plot_views_3x3(prepared, views, cfg, style, overlay)
        fig_jnd = plot_jnd_figure(group_jnd_by_view, views, style, view_colors=view_colors)
    elif layout == "abls_4x3":
        fig_main = plot_abls_4x3(prepared, views, cfg, style, overlay, view_colors=view_colors)
        fig_jnd = plot_jnd_figure(group_jnd_by_view, views, style, view_colors=view_colors)
    else:
        raise ValueError(f"Unknown layout='{layout}'. Use 'views_3x3' or 'abls_4x3'.")

    if show:
        plt.show()

    return dict(
        prepared=prepared,
        group_jnd_by_view=group_jnd_by_view,
        fig_main=fig_main,
        fig_jnd=fig_jnd,
    )
