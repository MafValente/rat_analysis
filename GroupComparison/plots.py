# analysis/groupcomparison/plots.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Helpers.DataHelpers as DataHelpers

from .config import PlotStyle, GroupComparisonConfig, JNDOverlaySpec


def _maybe_shift(x, cfg: GroupComparisonConfig):
    if not cfg.ild_shift_for_abl50:
        return np.asarray(x, dtype=float)
    return DataHelpers.shift_ILD_for_ABL50(x)


def style_axes(ax, style: PlotStyle, title=None, xlabel=None, ylabel=None, square=True):
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


def apply_50_tick_labels(ax, xlim=(-19, 19)):
    lo, hi = xlim
    xticks = sorted(set(ax.get_xticks()) | {-18, 18})

    # keep only ticks inside limits (this removes -20, 20, etc.)
    xticks = [x for x in xticks if lo <= x <= hi]

    ax.set_xticks(xticks)
    ax.set_xticklabels([("-50" if x == -18 else "50" if x == 18 else str(int(x))) for x in xticks])


def plot_rt_on_ax(
    ax,
    tables: dict,
    abl: int,
    color: str,
    cfg: GroupComparisonConfig,
    *,
    linestyle: str = "-",
    marker: str = "o",
    markerfacecolor=None,
):
    rt_group = tables["rt_group"]
    rt_per_subj = tables["rt_per_subj"]
    markerfacecolor = color if markerfacecolor is None else markerfacecolor

    if cfg.error_mode == "sem":
        sub = rt_group[rt_group["ABL"] == abl]
        if sub.empty:
            return
        x = _maybe_shift(sub["ILD"], cfg)
        ax.errorbar(
            x, sub["mean"], yerr=sub["sem"], fmt=marker, color=color,
            markerfacecolor=markerfacecolor, markeredgecolor=color,
            markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3,
        )
        return

    sub_ps = rt_per_subj[rt_per_subj["ABL"] == abl]
    for _, df_an in sub_ps.groupby("animal"):
        df_an = df_an.sort_values("ILD")
        ax.plot(_maybe_shift(df_an["ILD"], cfg), df_an["mean_rt"],
                color=color, alpha=0.35, linewidth=1.5, linestyle=linestyle)

    sub = rt_group[rt_group["ABL"] == abl].sort_values("ILD")
    if sub.empty:
        return
    x = _maybe_shift(sub["ILD"], cfg)
    ax.errorbar(
        x, sub["mean"], yerr=sub["sem"], fmt=marker, color=color,
        markerfacecolor=markerfacecolor, markeredgecolor=color,
        markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3,
    )
    ax.plot(x, sub["mean"], color=color, linewidth=2.5, linestyle=linestyle)


def plot_mt_on_ax(
    ax,
    tables: dict,
    abl: int,
    color: str,
    cfg: GroupComparisonConfig,
    *,
    linestyle: str = "-",
    marker: str = "o",
    markerfacecolor=None,
):
    mt_group = tables["mt_group"]
    mt_per_subj = tables["mt_per_subj"]
    markerfacecolor = color if markerfacecolor is None else markerfacecolor

    if cfg.error_mode == "sem":
        sub = mt_group[mt_group["ABL"] == abl]
        if sub.empty:
            return
        x = _maybe_shift(sub["ILD"], cfg)
        ax.errorbar(
            x, sub["mean"], yerr=sub["sem"], fmt=marker, color=color,
            markerfacecolor=markerfacecolor, markeredgecolor=color,
            markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3,
        )
        return

    sub_ps = mt_per_subj[mt_per_subj["ABL"] == abl]
    for _, df_an in sub_ps.groupby("animal"):
        df_an = df_an.sort_values("ILD")
        ax.plot(_maybe_shift(df_an["ILD"], cfg), df_an["mean_mt"],
                color=color, alpha=0.35, linewidth=1.5, linestyle=linestyle)

    sub = mt_group[mt_group["ABL"] == abl].sort_values("ILD")
    if sub.empty:
        return
    x = _maybe_shift(sub["ILD"], cfg)
    ax.errorbar(
        x, sub["mean"], yerr=sub["sem"], fmt=marker, color=color,
        markerfacecolor=markerfacecolor, markeredgecolor=color,
        markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3,
    )
    ax.plot(x, sub["mean"], color=color, linewidth=2.5, linestyle=linestyle)


def plot_psy_on_ax(
    ax,
    tables: dict,
    abl: int,
    color: str,
    cfg: GroupComparisonConfig,
    *,
    linestyle: str = "-",
    marker: str = "o",
    markerfacecolor=None,
):
    psy_group = tables["psy_group"]
    psy_indiv = tables["psy_indiv_curves"]
    psy_mean  = tables["psy_mean_fits"]
    markerfacecolor = color if markerfacecolor is None else markerfacecolor

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1)

    sub = psy_group[psy_group["ABL"] == abl]
    if not sub.empty:
        x_vals = _maybe_shift(sub["ILD"], cfg)
        ax.errorbar(x_vals, sub["mean"], yerr=sub["sem"],
                    fmt=marker, color=color, markersize=8.5,
                    markerfacecolor=markerfacecolor, markeredgecolor=color,
                    linewidth=0, elinewidth=1.5, capsize=3)

    if abl in set(cfg.skip_psy_fits):
        return

    if cfg.error_mode == "individuals":
        for (subject, abl_key), curve in psy_indiv.items():
            if abl_key != abl:
                continue
            ax.plot(_maybe_shift(curve["xx"], cfg), curve["yy"],
                    color=color, alpha=0.3, linewidth=1, linestyle=linestyle)

    mean_fit = psy_mean.get(abl)
    if mean_fit:
        ax.plot(_maybe_shift(mean_fit["xx"], cfg), mean_fit["yy"],
                color=color, linewidth=(3 if cfg.error_mode == "individuals" else 2), linestyle=linestyle)


def plot_psychometric_animals_plus_average(
    *,
    tables: dict,
    view_name: str,
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    n_cols: int = 3,
) -> plt.Figure:
    """
    One panel per animal plus a final average panel.

    Each panel shows psychometric points and fits by ABL.
    The last panel shows the group average across animals.
    """
    psy_points = tables["psy_points"]
    psy_group = tables["psy_group"]
    psy_indiv = tables["psy_indiv_curves"]
    psy_mean = tables["psy_mean_fits"]

    if psy_points is None or psy_points.empty:
        raise ValueError(f"No psychometric points available for view '{view_name}'.")

    animals = sorted(psy_points["subject"].dropna().astype(str).unique())
    panel_names = list(animals) + ["Average"]
    n_panels = len(panel_names)
    n_cols = max(1, min(int(n_cols), n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )

    all_abls = sorted(
        set(int(a) for a in psy_points["ABL"].dropna().unique())
        | set(int(a) for a in psy_group["ABL"].dropna().unique())
    )
    abl_colors = _default_abl_color_map_from_abls(all_abls)

    def _draw_panel(
        ax,
        *,
        points_df: pd.DataFrame,
        mean_df: pd.DataFrame | None,
        indiv_curves: dict | None,
        mean_fits: dict | None,
        panel_title: str,
        show_mean_errorbars: bool,
    ) -> None:
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylim(0, 1)

        for abl in all_abls:
            c = abl_colors.get(int(abl), "gray")

            sub = points_df[points_df["ABL"] == abl].sort_values("ILD")
            if not sub.empty:
                y_col = "PropLeft" if "PropLeft" in sub.columns else "mean"
                ax.scatter(
                    _maybe_shift(sub["ILD"], cfg),
                    sub[y_col],
                    color=c,
                    s=24,
                    alpha=0.95,
                    edgecolors="none",
                )

            if indiv_curves and cfg.error_mode == "individuals":
                for key, curve in indiv_curves.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        if int(key[1]) != int(abl):
                            continue
                    else:
                        continue
                    ax.plot(
                        _maybe_shift(curve["xx"], cfg),
                        curve["yy"],
                        color=c,
                        alpha=0.28,
                        linewidth=1.0,
                        linestyle="-",
                    )

            if mean_df is not None:
                subm = mean_df[mean_df["ABL"] == abl].sort_values("ILD")
                if show_mean_errorbars and not subm.empty:
                    ax.errorbar(
                        _maybe_shift(subm["ILD"], cfg),
                        subm["mean"],
                        yerr=subm["sem"],
                        fmt="o",
                        color=c,
                        markerfacecolor=c,
                        markeredgecolor=c,
                        markersize=5.5,
                        linewidth=0,
                        elinewidth=1.2,
                        capsize=2.5,
                    )

            mean_fit = (mean_fits or {}).get(int(abl))
            if mean_fit:
                ax.plot(
                    _maybe_shift(mean_fit["xx"], cfg),
                    mean_fit["yy"],
                    color=c,
                    linewidth=2.6,
                    linestyle="-",
                )

        ax.set_xlim(*cfg.xlim_sym)
        apply_50_tick_labels(ax, cfg.xlim_sym)
        style_axes(ax, style, panel_title, None, None, square=True)

    for idx, name in enumerate(panel_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        if name == "Average":
            _draw_panel(
                ax,
                points_df=psy_group,
                mean_df=psy_group,
                indiv_curves=None,
                mean_fits=psy_mean,
                panel_title=f"{view_name} - average",
                show_mean_errorbars=True,
            )
        else:
            points_df = psy_points[psy_points["subject"].astype(str) == name]
            indiv_curves = {
                key: curve
                for key, curve in psy_indiv.items()
                if isinstance(key, tuple) and str(key[0]) == name
            }
            _draw_panel(
                ax,
                points_df=points_df,
                mean_df=None,
                indiv_curves=indiv_curves,
                mean_fits=None,
                panel_title=name,
                show_mean_errorbars=False,
            )

        if col == 0:
            ax.set_ylabel("P(Left)", fontsize=style.label_fs, color="black")
        if row == n_rows - 1:
            ax.set_xlabel("ILD (dB)", fontsize=style.label_fs, color="black")

    for ax in axes.reshape(-1)[n_panels:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], color=abl_colors[abl], marker="o", linestyle="None")
        for abl in all_abls
    ]
    labels = [f"ABL {abl}" for abl in all_abls]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(6, len(handles)),
        fontsize=style.legend_fs,
        frameon=False,
    )
    fig.suptitle(f"{view_name} psychometrics by animal", fontsize=style.title_fs + 2, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def _animal_label(value) -> str:
    if isinstance(value, tuple) and len(value) >= 2:
        return f"{value[0]} {value[1]}"
    return str(value)


def _animal_sort_key(value):
    import re

    if isinstance(value, tuple) and len(value) >= 2:
        left = str(value[0])
        right = str(value[1])
        try:
            right_num = int(right)
        except Exception:
            right_num = 10**9
        return (left, right_num, right)

    s = str(value)
    m = re.search(r"(\d+)$", s)
    prefix = s[: m.start(1)] if m else s
    suffix = int(m.group(1)) if m else 10**9
    return (prefix, suffix, s)


def _compute_params_from_df(df: pd.DataFrame, *, skip_abls: Sequence[int] = (50,)) -> pd.DataFrame:
    from analysis import psychometric as Psychometric

    rows = []
    if df is None or df.empty:
        return pd.DataFrame(columns=["animal", "batch_name", "cohort", "line", "genotype", "dataset_key", "ABL", "slope_a", "bias_b", "lower_c", "upper_d"])

    skip_set = {int(a) for a in skip_abls} if skip_abls is not None else set()
    group_cols = ["batch_name", "animal"] if "batch_name" in df.columns else ["animal"]
    for subject_key, df_an in df.groupby(group_cols, sort=False):
        animal = subject_key if len(group_cols) == 1 else tuple(subject_key)
        results = Psychometric.compute_psychometrics_by_ABL(df_an, model="my_psycho")
        for abl, res in results.items():
            if int(abl) in skip_set:
                continue
            pars = res.get("pars")
            if pars is None or len(pars) < 4:
                continue
            pars = np.asarray(pars[:4], dtype=float)
            if not np.all(np.isfinite(pars)):
                continue
            rows.append(
                {
                    "animal": animal,
                    "batch_name": df_an["batch_name"].iloc[0] if "batch_name" in df_an.columns and not df_an.empty else pd.NA,
                    "cohort": df_an["cohort"].iloc[0] if "cohort" in df_an.columns and not df_an.empty else pd.NA,
                    "line": df_an["line"].iloc[0] if "line" in df_an.columns and not df_an.empty else pd.NA,
                    "genotype": df_an["genotype"].iloc[0] if "genotype" in df_an.columns and not df_an.empty else pd.NA,
                    "dataset_key": df_an["dataset_key"].iloc[0] if "dataset_key" in df_an.columns and not df_an.empty else pd.NA,
                    "ABL": int(abl),
                    "slope_a": float(pars[0]),
                    "bias_b": float(pars[1]),
                    "lower_c": float(pars[2]),
                    "upper_d": float(pars[3]),
                }
            )
    return pd.DataFrame(rows, columns=["animal", "batch_name", "cohort", "line", "genotype", "dataset_key", "ABL", "slope_a", "bias_b", "lower_c", "upper_d"])


def _compute_jnd_from_df(
    df: pd.DataFrame,
    *,
    skip_abl: int = 50,
    allowed_abls: Sequence[int] | None = None,
) -> pd.DataFrame:
    from analysis import psychometric as Psychometric

    if df is None or df.empty:
        return pd.DataFrame(columns=["animal", "animal_key", "ABL", "JND"])

    df = df.copy()
    if "ABL" in df.columns:
        df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
        df = df[df["ABL"].notna()].copy()
        df["ABL"] = df["ABL"].astype(int)
    if allowed_abls is not None:
        allowed_set = {int(a) for a in allowed_abls}
        df = df[df["ABL"].isin(allowed_set)].copy()
    if skip_abl is not None:
        df = df[df["ABL"] != int(skip_abl)].copy()

    if df.empty:
        return pd.DataFrame(columns=["animal", "animal_key", "ABL", "JND"])

    subject_cols = ["batch_name", "animal"] if "batch_name" in df.columns else ["animal"]
    rows = []
    for subject_key, df_subj in df.groupby(subject_cols, sort=False):
        animal = subject_key if len(subject_cols) == 1 else tuple(subject_key)
        animal_key = _animal_key(animal)
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_abl)
        if jnd_df is None or jnd_df.empty:
            continue
        for _, r in jnd_df.iterrows():
            try:
                rows.append(
                    {
                        "animal": animal,
                        "animal_key": animal_key,
                        "ABL": int(r["ABL"]),
                        "JND": float(r["JND"]),
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows, columns=["animal", "animal_key", "ABL", "JND"])


def plot_psychometric_params_all_views(
    *,
    prepared: dict,
    views: Sequence,
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    overlay_data: dict | None = None,
) -> plt.Figure:
    """
    Four panels, one per psychometric parameter, combined across all selected views.

    X-axis order:
      old overlay animals first, then each selected dataset block in the order supplied.
    Within each current dataset block, animals are ordered by genotype and animal id.
    """
    current_frames = []
    block_order = []
    seen_blocks = set()
    for v in views:
        params_df = prepared[v.name].get("psy_params", pd.DataFrame()).copy()
        if params_df.empty:
            continue
        skip_abls = tuple(int(a) for a in cfg.skip_psy_fits) if cfg.skip_psy_fits is not None else ()
        params_df = params_df[~params_df["ABL"].isin(skip_abls)].copy()
        if params_df.empty:
            continue
        params_df["view_name"] = v.name
        params_df["source"] = "current"
        params_df["block"] = params_df["dataset_key"].fillna(v.name).astype(str)
        current_frames.append(params_df)
        for block in params_df["block"].dropna().astype(str).unique():
            if block not in seen_blocks:
                seen_blocks.add(block)
                block_order.append(block)

    current_df = pd.concat(current_frames, ignore_index=True, sort=False) if current_frames else pd.DataFrame()
    if current_df.empty:
        raise ValueError("No psychometric parameters available for the selected views.")

    old_df = pd.DataFrame()
    old_order = []
    old_label_map = {}
    if overlay_data is not None and overlay_data.get("merged_valid") is not None:
        old_df = _compute_params_from_df(overlay_data["merged_valid"], skip_abls=tuple(int(a) for a in cfg.skip_psy_fits or ()))
        if not old_df.empty:
            old_df = old_df.copy()
            old_df["source"] = "old"
            old_df["block"] = "old"
        if overlay_data.get("unique_animal_identifiers"):
            for entry in overlay_data["unique_animal_identifiers"]:
                old_order.append(entry)
                old_label_map[entry] = entry
        elif not old_df.empty:
            old_order = list(old_df["animal"].dropna().unique())

    current_abls = sorted(int(a) for a in current_df["ABL"].dropna().unique())
    old_allowed_abls = [abl for abl in current_abls if abl != 30]
    jnd_common_abls = [abl for abl in current_abls if abl in (20, 40, 60)]
    if not jnd_common_abls:
        jnd_common_abls = [abl for abl in current_abls if abl != 50]
    if not old_df.empty:
        old_df = old_df[old_df["ABL"].isin(old_allowed_abls)].copy()
        old_df["source"] = "old"
        old_df["block"] = "old"

    combined = pd.concat([old_df, current_df], ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("No psychometric parameters available after combining old overlay and current views.")

    ordered_blocks = (["old"] if not old_df.empty else []) + block_order
    genotype_order = {"wt": 0, "het": 1, "hom": 2}
    genotype_markers = {"wt": "o", "het": "s", "hom": "^"}
    genotype_display = {"wt": "WT", "het": "HET", "hom": "HOM"}

    ordered_animals = []
    boundary_positions = []
    x_pos = 0.0
    for block in ordered_blocks:
        rows = combined[combined["block"].astype(str) == block].copy()
        if rows.empty:
            continue
        if block == "old" and old_order:
            animals = [a for a in old_order if a in set(rows["animal"])]
        else:
            rows["genotype_rank"] = rows["genotype"].astype(str).map(genotype_order).fillna(99).astype(int)
            animal_rows = (
                rows[["animal", "genotype_rank"]]
                .drop_duplicates()
                .sort_values(["genotype_rank", "animal"], kind="stable")
            )
            animals = animal_rows["animal"].tolist()
        start = x_pos
        for animal in animals:
            ordered_animals.append((block, animal, x_pos))
            x_pos += 1.0
        if block != ordered_blocks[-1]:
            boundary_positions.append(x_pos - 0.5)
            x_pos += 1.0

    if not ordered_animals:
        raise ValueError("No animal rows available for the selected views.")

    all_abls = sorted(int(a) for a in combined["ABL"].dropna().unique())
    abl_colors = _default_abl_color_map_from_abls(all_abls)
    params = [
        ("slope_a", "Slope (a)"),
        ("bias_b", "Bias (b)"),
        ("lower_c", "Lower (c)"),
        ("upper_d", "Upper (d)"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(max(18.0, 0.55 * len(ordered_animals)), 12.2),
        squeeze=False,
        sharex=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.22},
    )
    axes = axes.ravel()

    for ax, (param_col, title) in zip(axes, params):
        for block, animal, xpos in ordered_animals:
            sub = combined[(combined["block"].astype(str) == block) & (combined["animal"] == animal)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("ABL")
            genotype = str(sub["genotype"].iloc[0]) if "genotype" in sub.columns and sub["genotype"].notna().any() else "wt"
            marker = genotype_markers.get(genotype, "o")
            for _, row in sub.iterrows():
                abl = int(row["ABL"])
                color = abl_colors.get(abl, "gray")
                face = color
                edge = color
                alpha = 0.55 if block == "old" else 0.9
                linewidth = 1.2 if block == "old" else 0.8
                ax.scatter(
                    xpos,
                    row[param_col],
                    s=78,
                    marker=marker if block != "old" else "o",
                    facecolors=face,
                    edgecolors=edge,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=3,
                )

        for xpos in boundary_positions:
            ax.axvline(xpos, color="0.7", linestyle="--", linewidth=1.0, zorder=0)

        ax.set_title(title, fontsize=style.title_fs, pad=style.title_pad)
        ax.set_ylabel("Parameter value", fontsize=style.label_fs, color="black")
        ax.grid(True, axis="x", linestyle=":", alpha=0.25)
        ax.set_xlim(-0.8, max(x for _, _, x in ordered_animals) + 0.8)
        ax.tick_params(axis="y", labelsize=style.tick_fs)

    tick_positions = [x for _, _, x in ordered_animals]
    tick_labels = [
        _animal_label(old_label_map.get(animal, animal)) if block == "old" else _animal_label(animal)
        for block, animal, _ in ordered_animals
    ]
    for ax in axes[2:]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=max(8, style.tick_fs - 3))
        ax.set_xlabel("Animal", fontsize=style.label_fs, color="black")
    for ax in axes[:2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([])

    abl_handles = [
        plt.Line2D([], [], marker="o", color="black", markerfacecolor=abl_colors[abl], markersize=7, linestyle="None")
        for abl in all_abls
    ]
    abl_labels = [f"ABL {abl}" for abl in all_abls]
    genotype_handles = [
        plt.Line2D([], [], marker=genotype_markers[g], color="black", markerfacecolor="black", markersize=7, linestyle="None")
        for g in ("wt", "het", "hom")
    ]
    genotype_labels = [genotype_display[g] for g in ("wt", "het", "hom")]
    source_handles = [
        plt.Line2D([], [], marker="o", color="black", markerfacecolor="none", markersize=7, linestyle="None", markeredgewidth=1.2),
        plt.Line2D([], [], marker="o", color="black", markerfacecolor="black", markersize=7, linestyle="None"),
    ]
    source_labels = ["old overlay", "current cohorts"]
    fig.legend(
        abl_handles + genotype_handles + source_handles,
        abl_labels + genotype_labels + source_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(6, len(abl_labels) + len(genotype_labels) + 2),
        fontsize=style.legend_fs,
        frameon=False,
    )
    fig.suptitle("Psychometric parameters by animal", fontsize=style.title_fs + 2, y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    return fig


def _compute_animal_summary_metrics_from_df(
    df: pd.DataFrame,
    *,
    skip_abls: Sequence[int] = (50,),
    allowed_abls: Sequence[int] | None = None,
    included_abort_types: Sequence[str] | None = None,
    speed_distance_cm: float = 1.0,
    led0_correction_fraction: float | None = None,
) -> pd.DataFrame:
    rows = []
    abort_types = tuple(included_abort_types) if included_abort_types is not None else ()
    abort_cols = [f"abort_{_abort_type_key(a)}" for a in abort_types]
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["animal", "animal_key", "cohort", "line", "genotype", "dataset_key", "ABL", "accuracy", "bias", "rt", "speed", "abort_rate", *abort_cols]
        )

    df = df.copy()
    if "ABL" in df.columns:
        df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
        df = df[df["ABL"].notna()].copy()
        df["ABL"] = df["ABL"].astype(int)
    if allowed_abls is not None:
        allowed_set = {int(a) for a in allowed_abls}
        df = df[df["ABL"].isin(allowed_set)].copy()
    skip_set = {int(a) for a in skip_abls} if skip_abls is not None else set()
    if skip_set:
        df = df[~df["ABL"].isin(skip_set)].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["animal", "animal_key", "cohort", "line", "genotype", "dataset_key", "ABL", "accuracy", "bias", "rt", "speed", "abort_rate", *abort_cols]
        )

    subject_cols = ["batch_name", "animal"] if "batch_name" in df.columns else ["animal"]
    for subject_key, df_an in df.groupby(subject_cols, sort=False):
        animal = subject_key if len(subject_cols) == 1 else tuple(subject_key)
        animal_key = _animal_key(animal)
        valid_trials_per_session = np.nan
        estimated_valid_trials_total = np.nan
        if "success" in df_an.columns:
            valid_trials = df_an[df_an["success"] != 0].copy()
            if not valid_trials.empty:
                if "session" in valid_trials.columns:
                    valid_counts = valid_trials.groupby("session", observed=False).size().astype(float)
                    if led0_correction_fraction is not None and "LED_trial" in valid_trials.columns:
                        session_led0 = (
                            valid_trials.groupby("session", observed=False)["LED_trial"]
                            .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna())
                        )
                        for session_id in valid_counts.index:
                            led_vals = session_led0.get(session_id, pd.Series(dtype=float))
                            if not led_vals.empty and (led_vals == 0).all():
                                valid_counts.loc[session_id] = valid_counts.loc[session_id] * (1.0 + float(led0_correction_fraction))
                    if not valid_counts.empty:
                        valid_trials_per_session = float(valid_counts.mean())
                        estimated_total = 0.0
                        for session_id, session_trials in valid_trials.groupby("session", observed=False):
                            session_count = float(len(session_trials))
                            if "LED_trial" in session_trials.columns:
                                led_vals = pd.to_numeric(session_trials["LED_trial"], errors="coerce")
                                led_vals = led_vals.dropna()
                                if (
                                    led0_correction_fraction is not None
                                    and not led_vals.empty
                                    and (led_vals == 0).all()
                                ):
                                    session_count = session_count * (1.0 + float(led0_correction_fraction))
                            estimated_total += session_count
                        estimated_valid_trials_total = float(estimated_total)
                else:
                    valid_trials_per_session = float(len(valid_trials))
                    session_count = float(len(valid_trials))
                    if "LED_trial" in valid_trials.columns:
                        led_vals = pd.to_numeric(valid_trials["LED_trial"], errors="coerce").dropna()
                        if (
                            led0_correction_fraction is not None
                            and not led_vals.empty
                            and (led_vals == 0).all()
                        ):
                            session_count = session_count * (1.0 + float(led0_correction_fraction))
                    estimated_valid_trials_total = float(session_count)
        for abl, df_abl in df_an.groupby("ABL", sort=False):
            accuracy = np.nan
            valid = df_abl[df_abl["success"] != 0] if "success" in df_abl.columns else pd.DataFrame()
            if not valid.empty and "success" in valid.columns:
                accuracy = float((valid["success"] == 1).mean())

            rt = np.nan
            if {"success", "timed_rt"}.issubset(df_abl.columns):
                rt_df = df_abl[(df_abl["success"] == 1) & df_abl["timed_rt"].notna() & (df_abl["timed_rt"] <= 1.2)]
                if not rt_df.empty:
                    rt = float(rt_df["timed_rt"].mean())

            speed = np.nan
            mt_series = _combined_mt_series(df_abl)
            if mt_series is not None and "success" in df_abl.columns:
                mt_df = df_abl[(df_abl["success"] == 1) & mt_series.notna() & (mt_series > 0)]
                if not mt_df.empty:
                    mt_mean = float(pd.to_numeric(mt_series.loc[mt_df.index], errors="coerce").mean())
                    if mt_mean > 0 and speed_distance_cm > 0:
                        speed = float(speed_distance_cm / mt_mean)

            bias = np.nan
            if "abort_type" in df_abl.columns:
                bias_df = df_abl[df_abl["abort_type"] != "CNP"].copy()
            else:
                bias_df = df_abl.copy()
            if not bias_df.empty:
                try:
                    bias = float(DataHelpers.compute_bias(bias_df))
                except Exception:
                    bias = np.nan

            abort_rate = np.nan
            abort_type_rates = {f"abort_{_abort_type_key(a)}": np.nan for a in abort_types}
            if "success" in df_abl.columns:
                abort_rows = df_abl[df_abl["success"] == 0].copy()
                if not abort_rows.empty:
                    abort_rate = float(len(abort_rows) / len(df_abl))
                    if "abort_type" in abort_rows.columns and included_abort_types is not None:
                        for abort_type in abort_types:
                            col = f"abort_{_abort_type_key(abort_type)}"
                            abort_type_rates[col] = float((abort_rows["abort_type"] == abort_type).sum() / len(df_abl))

            rows.append(
                {
                    "animal": animal,
                    "animal_key": animal_key,
                    "cohort": df_abl["cohort"].iloc[0] if "cohort" in df_abl.columns and not df_abl.empty else pd.NA,
                    "line": df_abl["line"].iloc[0] if "line" in df_abl.columns and not df_abl.empty else pd.NA,
                    "genotype": df_abl["genotype"].iloc[0] if "genotype" in df_abl.columns and not df_abl.empty else pd.NA,
                    "dataset_key": df_abl["dataset_key"].iloc[0] if "dataset_key" in df_abl.columns and not df_abl.empty else pd.NA,
                    "ABL": int(abl),
                    "accuracy": accuracy,
                    "bias": bias,
                    "rt": rt,
                    "speed": speed,
                    "valid_trials_per_session": valid_trials_per_session,
                    "valid_trials_total_est": estimated_valid_trials_total,
                    "abort_rate": abort_rate,
                    **abort_type_rates,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["animal", "animal_key", "cohort", "line", "genotype", "dataset_key", "ABL", "accuracy", "bias", "rt", "speed", "valid_trials_per_session", "valid_trials_total_est", "abort_rate", *abort_cols],
    )


def _compute_abort_type_fractions_from_df(
    df: pd.DataFrame,
    *,
    skip_abls: Sequence[int] = (50,),
    allowed_abls: Sequence[int] | None = None,
    abort_types: Sequence[str] = ("Fixation", "RT-", "MT+"),
) -> pd.DataFrame:
    rows = []
    abort_cols = [f"abort_{_abort_type_key(a)}" for a in abort_types]
    if df is None or df.empty:
        return pd.DataFrame(columns=["animal", "animal_key", "cohort", "line", "genotype", "dataset_key", "ABL", *abort_cols])

    df = df.copy()
    if "ABL" in df.columns:
        df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
        df = df[df["ABL"].notna()].copy()
        df["ABL"] = df["ABL"].astype(int)
    if allowed_abls is not None:
        allowed_set = {int(a) for a in allowed_abls}
        df = df[df["ABL"].isin(allowed_set)].copy()
    skip_set = {int(a) for a in skip_abls} if skip_abls is not None else set()
    if skip_set:
        df = df[~df["ABL"].isin(skip_set)].copy()

    if df.empty:
        return pd.DataFrame(columns=["animal", "animal_key", "cohort", "line", "genotype", "dataset_key", "ABL", *abort_cols])

    subject_cols = ["batch_name", "animal"] if "batch_name" in df.columns else ["animal"]
    for subject_key, df_an in df.groupby(subject_cols, sort=False):
        animal = subject_key if len(subject_cols) == 1 else tuple(subject_key)
        animal_key = _animal_key(animal)
        for abl, df_abl in df_an.groupby("ABL", sort=False):
            row = {
                "animal": animal,
                "animal_key": animal_key,
                "cohort": df_abl["cohort"].iloc[0] if "cohort" in df_abl.columns and not df_abl.empty else pd.NA,
                "line": df_abl["line"].iloc[0] if "line" in df_abl.columns and not df_abl.empty else pd.NA,
                "genotype": df_abl["genotype"].iloc[0] if "genotype" in df_abl.columns and not df_abl.empty else pd.NA,
                "dataset_key": df_abl["dataset_key"].iloc[0] if "dataset_key" in df_abl.columns and not df_abl.empty else pd.NA,
                "ABL": int(abl),
            }
            active = df_abl[df_abl["success"] == 0].copy() if "success" in df_abl.columns else pd.DataFrame()
            for abort_type, col in zip(abort_types, abort_cols):
                row[col] = np.nan
            if not active.empty and "abort_type" in active.columns:
                for abort_type, col in zip(abort_types, abort_cols):
                    row[col] = float((active["abort_type"] == abort_type).sum() / len(df_abl))
            rows.append(row)

    return pd.DataFrame(rows, columns=["animal", "animal_key", "cohort", "line", "genotype", "dataset_key", "ABL", *abort_cols])


def plot_summary_metrics_all_views(
    *,
    df_filtered: pd.DataFrame,
    prepared: dict,
    views: Sequence,
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    overlay: object | None = None,
    jnd_indiv_by_view: dict | None = None,
    jnd_overlay: object | None = None,
    mode: str = "core",
) -> plt.Figure:
    """
    Summary metrics by animal.
    mode="core": accuracy, bias, RT, speed and JND.
    mode="aborts": fixation / RT- / MT+ abort fractions.
    """
    current_frames = []
    current_jnd_frames = []
    block_order = []
    seen_blocks = set()
    view_block_map = {}
    skip_abls = tuple(int(a) for a in cfg.skip_psy_fits) if cfg.skip_psy_fits is not None else ()
    current_abls = []
    jnd_common_abls = []

    for v in views:
        tables_v = prepared[v.name]
        df_v = tables_v.get("df_view", v.selector(df_filtered)).copy()
        if df_v.empty:
            continue
        params_df = _compute_animal_summary_metrics_from_df(
            df_v,
            skip_abls=skip_abls,
            included_abort_types=cfg.summary_abort_types,
            speed_distance_cm=cfg.current_speed_distance_cm,
            led0_correction_fraction=None,
        )
        if params_df.empty:
            continue
        params_df["view_name"] = v.name
        params_df["source"] = "current"
        params_df["animal_key"] = params_df["animal"].map(_animal_key)
        block_value = str(params_df["dataset_key"].dropna().astype(str).iloc[0]) if params_df["dataset_key"].notna().any() else v.name
        params_df["block"] = block_value
        view_block_map[v.name] = block_value
        current_frames.append(params_df)
        for block in params_df["block"].dropna().astype(str).unique():
            if block not in seen_blocks:
                seen_blocks.add(block)
                block_order.append(block)

        if jnd_indiv_by_view is not None:
            dfj = jnd_indiv_by_view.get(v.name)
            if dfj is not None and not dfj.empty:
                dfj = dfj.copy()
                dfj["block"] = block_value
                dfj["source"] = "current"
                if "subject" in dfj.columns:
                    dfj["animal_key"] = dfj["subject"].map(_animal_key)
                current_jnd_frames.append(dfj)

    current_df = pd.concat(current_frames, ignore_index=True, sort=False) if current_frames else pd.DataFrame()
    if current_df.empty:
        raise ValueError("No summary metrics available for the selected views.")
    if "animal_key" not in current_df.columns:
        current_df["animal_key"] = current_df["animal"].map(_animal_key)
    current_abls = sorted(int(a) for a in current_df["ABL"].dropna().unique())
    jnd_common_abls = [abl for abl in (20, 40, 60) if abl in current_abls]
    if not jnd_common_abls:
        jnd_common_abls = [abl for abl in current_abls if abl != 50]
    current_jnd_df = pd.concat(current_jnd_frames, ignore_index=True, sort=False) if current_jnd_frames else pd.DataFrame()
    if not current_jnd_df.empty and "animal_key" in current_jnd_df.columns:
        current_genotype_map = (
            current_df[["block", "animal_key", "animal", "genotype"]]
            .drop_duplicates(subset=["block", "animal_key"])
            .copy()
        )
        current_jnd_df = current_jnd_df.merge(
            current_genotype_map[["block", "animal_key", "genotype"]],
            on=["block", "animal_key"],
            how="left",
        )
    if not current_jnd_df.empty:
        current_jnd_df = current_jnd_df[current_jnd_df["ABL"].isin(jnd_common_abls)].copy()
        current_jnd_df = current_jnd_df.drop_duplicates(subset=["block", "animal_key", "ABL"], keep="first").copy()

    old_allowed_abls = [abl for abl in current_abls if abl != 30]

    old_df = pd.DataFrame()
    old_jnd_df = pd.DataFrame()
    old_order = []
    old_label_map = {}
    overlay_data = getattr(overlay, "makefig1_data", None)
    overlay_chrono = getattr(overlay, "makefig1_chrono", None)
    overlay_jnd = getattr(jnd_overlay, "old_jnd_data", None) if jnd_overlay is not None else None

    if overlay_data is not None and overlay_data.get("merged_valid") is not None:
        old_df = _compute_animal_summary_metrics_from_df(
            overlay_data["merged_valid"],
            skip_abls=skip_abls,
            allowed_abls=old_allowed_abls,
            included_abort_types=cfg.summary_abort_types,
            speed_distance_cm=cfg.overlay_speed_distance_cm,
            led0_correction_fraction=0.1,
        )
        if not old_df.empty:
            old_df = old_df.copy()
            old_df["source"] = "old"
            old_df["block"] = "old"
            old_df["animal_key"] = old_df["animal"].map(_animal_key)
        if overlay_data.get("unique_animal_identifiers"):
            for entry in overlay_data["unique_animal_identifiers"]:
                old_order.append(entry)
                old_label_map[entry] = entry
        elif not old_df.empty:
            old_order = list(old_df["animal"].dropna().unique())

    if overlay_jnd is not None:
        old_jnd_rows = []
        old_jnds = overlay_jnd.get("jnds", {}) if isinstance(overlay_jnd, dict) else {}
        old_animals = overlay_jnd.get("animals_with_mean", []) if isinstance(overlay_jnd, dict) else []
        if not old_animals:
            old_animals = old_order
        for abl, animal_map in old_jnds.items():
            try:
                abl_i = int(abl)
            except Exception:
                continue
            if jnd_common_abls and abl_i not in jnd_common_abls:
                continue
            if not isinstance(animal_map, dict):
                continue
            for animal in old_animals:
                jnd_val = animal_map.get(animal, animal_map.get(_animal_key(animal)))
                if jnd_val is None:
                    continue
                try:
                    old_jnd_rows.append({
                        "animal": animal,
                        "animal_key": _animal_key(animal),
                        "ABL": abl_i,
                        "JND": float(jnd_val),
                    })
                except Exception:
                    continue
                if animal not in old_label_map:
                    old_label_map[animal] = animal
                if animal not in old_order:
                    old_order.append(animal)
        if old_jnd_rows:
            old_jnd_df = pd.DataFrame(old_jnd_rows)
            old_jnd_df["source"] = "old"
            old_jnd_df["block"] = "old"
    if not old_jnd_df.empty and not old_df.empty:
        old_genotype_map = old_df[["animal_key", "animal", "genotype"]].drop_duplicates(subset=["animal_key"]).copy()
        old_jnd_df = old_jnd_df.merge(
            old_genotype_map[["animal_key", "genotype"]],
            on="animal_key",
            how="left",
        )
    if not old_jnd_df.empty:
        old_jnd_df = old_jnd_df[old_jnd_df["ABL"].isin(jnd_common_abls)].copy()
        old_jnd_df = old_jnd_df.drop_duplicates(subset=["animal_key", "ABL"], keep="first").copy()

    if old_jnd_df.empty and overlay_data is not None and overlay_data.get("merged_valid") is not None:
        old_jnd_df = _compute_jnd_from_df(
            overlay_data["merged_valid"],
            skip_abl=skip_abls[0] if len(skip_abls) == 1 else 50,
            allowed_abls=jnd_common_abls,
        )
        if not old_jnd_df.empty:
            old_jnd_df["source"] = "old"
            old_jnd_df["block"] = "old"
            if old_order:
                pass
            else:
                old_order = list(old_jnd_df["animal"].dropna().unique())
            if not old_df.empty:
                old_genotype_map = old_df[["animal_key", "animal", "genotype"]].drop_duplicates(subset=["animal_key"]).copy()
                old_jnd_df = old_jnd_df.merge(
                    old_genotype_map[["animal_key", "genotype"]],
                    on="animal_key",
                    how="left",
                )
            old_jnd_df = old_jnd_df[old_jnd_df["ABL"].isin(jnd_common_abls)].copy()
            old_jnd_df = old_jnd_df.drop_duplicates(subset=["animal_key", "ABL"], keep="first").copy()

    if overlay_chrono is not None and "all_chrono_data_df" in overlay_chrono and not old_df.empty:
        chrono_df = overlay_chrono["all_chrono_data_df"].copy()
        if "ABL" in chrono_df.columns:
            chrono_df["ABL"] = pd.to_numeric(chrono_df["ABL"], errors="coerce")
            chrono_df = chrono_df[chrono_df["ABL"].notna()].copy()
            chrono_df["ABL"] = chrono_df["ABL"].astype(int)
        chrono_df = chrono_df[chrono_df["ABL"].isin(old_allowed_abls)].copy()
        if not chrono_df.empty and {"batch_name", "animal_id", "mean"}.issubset(chrono_df.columns):
            old_rt = (
                chrono_df.groupby(["batch_name", "animal_id", "ABL"], sort=False)["mean"]
                .mean()
                .reset_index()
                .rename(columns={"animal_id": "animal", "mean": "rt"})
            )
            old_rt["animal"] = list(zip(old_rt["batch_name"], old_rt["animal"]))
            old_rt = old_rt[["animal", "ABL", "rt"]].copy()
            old_df = old_df.drop(columns=["rt"], errors="ignore").merge(old_rt, on=["animal", "ABL"], how="left")

    if overlay_data is not None and overlay_data.get("merged_valid") is not None and not old_df.empty:
        # Keep a hook for a future dedicated bias overlay pickle.
        # Until that exists, keep the bias already computed from merged_valid.
        overlay_bias = getattr(overlay, "makefig1_bias", None)
        if overlay_bias is not None:
            old_bias = _compute_params_from_df(overlay_bias["merged_valid"], skip_abls=skip_abls)
            if not old_bias.empty:
                old_bias = old_bias[["animal", "ABL", "bias_b"]].rename(columns={"bias_b": "bias"})
                old_df = old_df.drop(columns=["bias"], errors="ignore").merge(old_bias, on=["animal", "ABL"], how="left", suffixes=("", "_bias"))
                if "bias_bias" in old_df.columns:
                    old_df["bias"] = old_df["bias_bias"].combine_first(old_df.get("bias"))
                old_df = old_df.drop(columns=["bias_bias"], errors="ignore")

    if not old_df.empty and "animal_key" not in old_df.columns:
        old_df["animal_key"] = old_df["animal"].map(_animal_key)
    combined = pd.concat([old_df, current_df], ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("No summary metrics available after combining old overlay and current views.")

    ordered_blocks = (["old"] if not old_df.empty else []) + block_order
    genotype_order = {"wt": 0, "het": 1, "hom": 2}
    genotype_markers = {"wt": "o", "het": "s", "hom": "^"}
    genotype_display = {"wt": "WT", "het": "HET", "hom": "HOM"}

    ordered_animals = []
    boundary_positions = []
    x_pos = 0.0
    for block in ordered_blocks:
        rows = combined[combined["block"].astype(str) == block].copy()
        if rows.empty:
            continue
        if block == "old" and old_order:
            animals = [a for a in old_order if _animal_key(a) in set(rows["animal_key"].astype(str))]
        else:
            rows["genotype_rank"] = rows["genotype"].astype(str).map(genotype_order).fillna(99).astype(int)
            animal_rows = (
                rows[["animal", "animal_key", "genotype_rank"]]
                .drop_duplicates()
                .sort_values(["genotype_rank", "animal_key"], kind="stable")
            )
            animals = animal_rows["animal"].tolist()
        for animal in animals:
            ordered_animals.append((block, animal, x_pos))
            x_pos += 1.0
        if block != ordered_blocks[-1]:
            boundary_positions.append(x_pos - 0.5)
            x_pos += 1.0

    if not ordered_animals:
        raise ValueError("No animal rows available for the selected views.")

    all_abls = sorted(int(a) for a in combined["ABL"].dropna().unique())
    abl_colors = _default_abl_color_map_from_abls(all_abls)
    if mode == "core":
        panels = [
            ("accuracy", "Accuracy"),
            ("bias", "Bias"),
            ("rt", "RT (s)"),
            ("speed", "Speed (cm/s)"),
            ("valid_trials_per_session", "Valid trials/session"),
            ("jnd", "JND (dB)"),
        ]
        fig, axes = plt.subplots(
            3,
            2,
            figsize=(max(18.0, 0.55 * len(ordered_animals)), 16.0),
            squeeze=False,
            sharex=True,
            gridspec_kw={"hspace": 0.38, "wspace": 0.22},
        )
    elif mode == "aborts":
        abort_types = [a for a in ("Fixation", "RT-", "MT+") if a in tuple(cfg.summary_abort_types)]
        if not abort_types:
            abort_types = list(cfg.summary_abort_types[:3])
        panels = [(f"abort_{_abort_type_key(a)}", f"{a} aborts") for a in abort_types]
        fig, axes = plt.subplots(
            1,
            max(1, len(panels)),
            figsize=(max(18.0, 0.60 * len(ordered_animals)), 7.5),
            squeeze=False,
            sharex=True,
            gridspec_kw={"wspace": 0.22},
        )
    else:
        raise ValueError("mode must be 'core' or 'aborts'")
    axes = axes.ravel()

    for ax, (param_col, title) in zip(axes, panels):
        for block, animal, xpos in ordered_animals:
            animal_key = _animal_key(animal)
            sub = combined[
                (combined["block"].astype(str) == block)
                & (combined["animal_key"].astype(str) == animal_key)
            ].copy()
            if param_col == "jnd":
                sub = pd.DataFrame()
                if not old_jnd_df.empty and block == "old":
                    sub = old_jnd_df[old_jnd_df["animal_key"].astype(str) == animal_key].copy()
                elif not current_jnd_df.empty and block != "old":
                    sub = current_jnd_df[current_jnd_df["block"].astype(str) == block].copy()
                    if "animal_key" in sub.columns:
                        sub = sub[sub["animal_key"].astype(str) == animal_key].copy()
            elif param_col == "valid_trials_per_session":
                sub = sub.drop_duplicates(subset=["animal_key"]).copy()
            elif param_col.startswith("abort_"):
                sub = sub.reindex(columns=["animal", "animal_key", "ABL", "genotype", param_col]).copy()
            if sub.empty:
                continue
            sub = sub.sort_values("ABL")
            if param_col == "valid_trials_per_session":
                genotype = str(sub["genotype"].iloc[0]) if "genotype" in sub.columns and sub["genotype"].notna().any() else "wt"
            elif param_col.startswith("abort_"):
                genotype = str(sub["genotype"].iloc[0]) if "genotype" in sub.columns and sub["genotype"].notna().any() else "wt"
            else:
                genotype = str(sub["genotype"].iloc[0]) if "genotype" in sub.columns and sub["genotype"].notna().any() else "wt"
            marker = genotype_markers.get(genotype, "o")
            for _, row in sub.iterrows():
                abl = int(row["ABL"])
                color = "0.35" if param_col == "valid_trials_per_session" else abl_colors.get(abl, "gray")
                face = color
                if param_col == "valid_trials_per_session":
                    face = "none" if block == "old" else color
                alpha = 0.55 if block == "old" else 0.9
                linewidth = 1.2 if block == "old" else 0.8
                if param_col.startswith("abort_"):
                    value = row.get(param_col, np.nan)
                elif param_col == "valid_trials_per_session":
                    value = row.get(param_col, np.nan)
                else:
                    value = row["JND"] if param_col == "jnd" else row[param_col]
                ax.scatter(
                    xpos,
                    value,
                    s=78,
                    marker=marker if block != "old" else "o",
                    facecolors=face,
                    edgecolors=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=3,
                )

        for xpos in boundary_positions:
            ax.axvline(xpos, color="0.7", linestyle="--", linewidth=1.0, zorder=0)

        ax.set_title(title, fontsize=style.title_fs, pad=style.title_pad)
        if param_col == "jnd":
            ylabel = "JND (dB)"
        elif param_col == "speed":
            ylabel = "Speed (cm/s)"
        elif param_col == "valid_trials_per_session":
            ylabel = "Valid trials/session"
        elif param_col.startswith("abort_"):
            ylabel = "Fraction of trials"
        else:
            ylabel = "Value"
        ax.set_ylabel(ylabel, fontsize=style.label_fs, color="black")
        ax.grid(True, axis="x", linestyle=":", alpha=0.25)
        ax.set_xlim(-0.8, max(x for _, _, x in ordered_animals) + 0.8)
        if param_col == "valid_trials_per_session":
            ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=style.tick_fs)
        ax._panel_tick_positions = [x for _, _, x in ordered_animals]
        ax._panel_tick_labels = [
            _animal_label(old_label_map.get(animal, animal)) if block == "old" else _animal_label(animal)
            for block, animal, _ in ordered_animals
        ]

    if mode == "core":
        for ax in axes:
            ax.set_xticks(getattr(ax, "_panel_tick_positions", []))
            ax.set_xticklabels([])
            ax.set_xlabel("")
        if len(panels) < len(axes):
            for ax in axes[len(panels):]:
                ax.axis("off")
    else:
        for ax in axes:
            ax.set_xticks(getattr(ax, "_panel_tick_positions", []))
            ax.set_xticklabels([])
            ax.set_xlabel("")

    abl_handles = [
        plt.Line2D([], [], marker="o", color="black", markerfacecolor=abl_colors[abl], markersize=7, linestyle="None")
        for abl in all_abls
    ]
    abl_labels = [f"ABL {abl}" for abl in all_abls]
    genotype_handles = [
        plt.Line2D([], [], marker=genotype_markers[g], color="black", markerfacecolor="black", markersize=7, linestyle="None")
        for g in ("wt", "het", "hom")
    ]
    genotype_labels = [genotype_display[g] for g in ("wt", "het", "hom")]
    source_handles = [
        plt.Line2D([], [], marker="o", color="black", markerfacecolor="none", markersize=7, linestyle="None", markeredgewidth=1.2),
        plt.Line2D([], [], marker="o", color="black", markerfacecolor="black", markersize=7, linestyle="None"),
    ]
    source_labels = ["old overlay", "current cohorts"]
    fig.legend(
        abl_handles + genotype_handles + source_handles,
        abl_labels + genotype_labels + source_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=min(6, len(abl_labels) + len(genotype_labels) + 2),
        fontsize=style.legend_fs,
        frameon=False,
    )
    fig.suptitle("Summary metrics by animal", fontsize=style.title_fs + 2, y=0.995)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    return fig


def plot_psychometric_params_by_animal(
    *,
    tables: dict,
    view_name: str,
    cfg: GroupComparisonConfig,
    style: PlotStyle,
    overlay_data: dict | None = None,
) -> plt.Figure:
    """
    Four panels, one per psychometric parameter.
    X-axis: animal IDs, ordered as old overlay animals first, then current cohorts.
    """
    params_df = tables.get("psy_params", pd.DataFrame()).copy()
    if params_df.empty:
        raise ValueError(f"No psychometric parameters available for view '{view_name}'.")

    skip_abls = tuple(int(a) for a in cfg.skip_psy_fits) if cfg.skip_psy_fits is not None else ()
    params_df = params_df[~params_df["ABL"].isin(skip_abls)].copy()

    old_df = pd.DataFrame()
    old_order = []
    old_label_map = {}
    if overlay_data is not None and overlay_data.get("merged_valid") is not None:
        old_df = _compute_params_from_df(overlay_data["merged_valid"], skip_abls=skip_abls)
        if overlay_data.get("unique_animal_identifiers"):
            old_order = []
            for entry in overlay_data["unique_animal_identifiers"]:
                old_order.append(entry)
                old_label_map[entry] = entry
        else:
            old_order = list(old_df["animal"].dropna().unique())

    params_df["source"] = "current"
    if not old_df.empty:
        old_df = old_df.copy()
        old_df["source"] = "old"

    current_abls = sorted(int(a) for a in params_df["ABL"].dropna().unique())
    old_allowed_abls = [abl for abl in current_abls if abl != 30]
    if not old_df.empty:
        old_df = old_df[old_df["ABL"].isin(old_allowed_abls)].copy()
        old_df["source"] = "old"

    combined = pd.concat([old_df, params_df], ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError(f"No psychometric parameters available after filtering for view '{view_name}'.")

    # group ordering: old overlay first, then current cohorts/datasets
    current_group_col = None
    if "cohort" in params_df.columns and params_df["cohort"].notna().any():
        current_group_col = "cohort"
    elif "dataset_key" in params_df.columns and params_df["dataset_key"].notna().any():
        current_group_col = "dataset_key"

    current_groups = []
    if current_group_col is not None:
        current_groups = [str(c) for c in params_df[current_group_col].dropna().astype(str).unique()]
        current_groups = sorted(current_groups)
    else:
        current_groups = ["current"]

    ordered_groups = (["old"] if not old_df.empty else []) + current_groups

    ordered_animals = []
    boundary_positions = []
    x_pos = 0.0
    group_centers = {}
    group_kind_map = {}
    for group in ordered_groups:
        if group == "old":
            rows = old_df.copy()
            group_kind = "old"
        elif current_group_col is not None:
            rows = combined[(combined["source"] == "current") & (combined[current_group_col].astype(str) == group)].copy()
            group_kind = "current"
        else:
            rows = combined[combined["source"] == "current"].copy()
            group_kind = "current"
        if rows.empty:
            continue
        group_kind_map[group] = group_kind
        if group == "old" and old_order:
            animals = [a for a in old_order if a in set(rows["animal"])]
        else:
            animals = sorted(rows["animal"].dropna().unique(), key=_animal_sort_key)
        start = x_pos
        for animal in animals:
            ordered_animals.append((group, animal, x_pos))
            x_pos += 1.0
        group_centers[group] = (start + x_pos - 1.0) / 2 if animals else start
        if group != ordered_groups[-1]:
            boundary_positions.append(x_pos - 0.5)
            x_pos += 1.0

    if not ordered_animals:
        raise ValueError(f"No animal rows available for view '{view_name}'.")

    abl_colors = _default_abl_color_map_from_abls(sorted(int(a) for a in combined["ABL"].unique()))
    params = [
        ("slope_a", "Slope (a)"),
        ("bias_b", "Bias (b)"),
        ("lower_c", "Lower (c)"),
        ("upper_d", "Upper (d)"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(max(16.0, 0.55 * len(ordered_animals)), 12.0),
        squeeze=False,
        sharex=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.22},
    )
    axes = axes.ravel()

    for ax, (param_col, title) in zip(axes, params):
        for group, animal, xpos in ordered_animals:
            if group == "old":
                sub = old_df[old_df["animal"] == animal]
            else:
                sub = combined[(combined["source"] == "current") & (combined["animal"] == animal)]
            if sub.empty:
                continue
            sub = sub.sort_values("ABL")
            group_kind = group_kind_map.get(group, "current")
            for _, row in sub.iterrows():
                abl = int(row["ABL"])
                color = abl_colors.get(abl, "gray")
                face = color
                alpha = 0.55 if group == "old" else 0.9
                linewidth = 1.2 if group == "old" else 0.8
                ax.scatter(
                    xpos,
                    row[param_col],
                    s=78,
                    marker="o",
                    facecolors=face,
                    edgecolors=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=3,
                )

        for xpos in boundary_positions:
            ax.axvline(xpos, color="0.7", linestyle="--", linewidth=1.0, zorder=0)

        ax.set_title(title, fontsize=style.title_fs, pad=style.title_pad)
        ax.set_ylabel("Parameter value", fontsize=style.label_fs, color="black")
        ax.grid(True, axis="x", linestyle=":", alpha=0.25)
        ax.set_xlim(-0.8, max(x for _, _, x in ordered_animals) + 0.8)
        ax.tick_params(axis="y", labelsize=style.tick_fs)

    # bottom labels
    tick_positions = [x for _, _, x in ordered_animals]
    tick_labels = [
        _animal_label(old_label_map.get(animal, animal)) if group == "old" else _animal_label(animal)
        for group, animal, _ in ordered_animals
    ]
    for ax in axes[2:]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=max(8, style.tick_fs - 3))
        ax.set_xlabel("Animal", fontsize=style.label_fs, color="black")

    # top row: no tick labels
    for ax in axes[:2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([])

    # legend
    abl_handles = [
        plt.Line2D([], [], marker="o", color="black", markerfacecolor=abl_colors[abl], markersize=7, linestyle="None")
        for abl in sorted(int(a) for a in combined["ABL"].unique())
    ]
    abl_labels = [f"ABL {abl}" for abl in sorted(int(a) for a in combined["ABL"].unique())]
    source_handles = [
        plt.Line2D([], [], marker="o", color="black", markerfacecolor="none", markersize=7, linestyle="None", markeredgewidth=1.2),
        plt.Line2D([], [], marker="o", color="black", markerfacecolor="black", markersize=7, linestyle="None"),
    ]
    source_labels = ["old overlay", "current cohorts"]
    fig.legend(
        abl_handles + source_handles,
        abl_labels + source_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(6, len(abl_labels) + 2),
        fontsize=style.legend_fs,
        frameon=False,
    )
    fig.suptitle(f"{view_name} psychometric parameters by animal", fontsize=style.title_fs + 2, y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    return fig


# ---------------------------
# JND plotting
# ---------------------------

def _default_abl_color_map_from_abls(abls: Sequence[int]) -> Dict[int, str]:
    # match your usual convention if possible
    preferred = {20: "C0", 40: "C1", 50: "C2", 60: "C3"}
    out = {}
    i = 0
    for abl in sorted(set(int(a) for a in abls)):
        if abl in preferred:
            out[abl] = preferred[abl]
        else:
            out[abl] = f"C{i % 10}"
            i += 1
    return out


def _animal_key(value) -> str:
    if isinstance(value, tuple) and len(value) >= 2:
        return str(value[1]).strip()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0]).strip()
    return str(value).strip()


def _abort_type_key(value) -> str:
    return (
        str(value)
        .strip()
        .replace("+", "plus")
        .replace("-", "minus")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _pick_mt_column(df: pd.DataFrame) -> str | None:
    for col in ("timed_mt", "timed_MT", "MT"):
        if col in df.columns:
            return col
    return None


def _combined_mt_series(df: pd.DataFrame) -> pd.Series | None:
    mt_series = None
    for col in ("timed_mt", "timed_MT", "MT"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        mt_series = vals if mt_series is None else mt_series.combine_first(vals)
    return mt_series


def _speed_distance_cm_for_block(block: str, cfg: GroupComparisonConfig) -> float:
    return float(cfg.overlay_speed_distance_cm if str(block) == "old" else cfg.current_speed_distance_cm)


def plot_jnd_comparison_per_view(
    fig,
    axes,
    view_names: Sequence[str],
    jnd_indiv_by_view: Dict[str, "pd.DataFrame"],
    jnd_overlay: JNDOverlaySpec,
    style: PlotStyle,
):
    """
    Each view gets its own axis:
      - old: open circles at abl + old_x_shift
      - new: filled circles at abl + new_x_shift
    """
    # lazy import to avoid hard dependency in module import
    

    old = jnd_overlay.old_jnd_data
    old_abls = []
    if old is not None and "ABLS" in old:
        old_abls = [int(x) for x in old["ABLS"]]

    # pick abls from new data too
    new_abls = []
    for vn in view_names:
        dfj = jnd_indiv_by_view.get(vn)
        if dfj is not None and not dfj.empty:
            new_abls.extend([int(x) for x in dfj["ABL"].unique()])

    all_abls = sorted(set(old_abls) | set(new_abls))
    if jnd_overlay.abl_color_map is not None:
        abl_color_map = {int(k): v for k, v in jnd_overlay.abl_color_map.items()}
    else:
        abl_color_map = _default_abl_color_map_from_abls(all_abls)

    for ax, vn in zip(axes, view_names):
        # --- OLD dataset (open circles) ---
        if old is not None:
            old_jnds = old.get("jnds", {})
            old_animals = old.get("animals_with_mean", [])

            for abl in all_abls:
                if abl not in old_jnds:
                    continue
                c = abl_color_map.get(abl, "gray")
                for animal in old_animals:
                    if animal in old_jnds[abl]:
                        ax.scatter(
                            abl + jnd_overlay.old_x_shift,
                            old_jnds[abl][animal],
                            facecolors="none",
                            edgecolors=c,
                            s=60,
                            lw=1,
                            alpha=0.9,
                        )

        # --- NEW dataset (filled circles) ---
        df_new = jnd_indiv_by_view.get(vn, pd.DataFrame())
        if df_new is not None and not df_new.empty:
            for _, r in df_new.iterrows():
                abl = int(r["ABL"])
                c = abl_color_map.get(abl, "gray")
                ax.scatter(
                    abl + jnd_overlay.new_x_shift,
                    float(r["JND"]),
                    color=c,
                    s=55,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # style
        style_axes(ax, style, title=f"{vn} — JND", xlabel="ABL (dB)", ylabel="JND (dB)")
        ax.set_xticks(all_abls if all_abls else [])
        if all_abls:
            ax.set_xlim(min(all_abls) - 5, max(all_abls) + 1)
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# def add_group_jnd_inset(
#     ax_parent,
#     group_jnd_by_view: Dict[str, "pd.DataFrame"],
#     view_names: Sequence[str],
#     view_colors: Dict[str, str],
#     style: PlotStyle,
#     inset_rect=(0.72, 0.15, 0.33, 0.33),
#     x_offsets: Optional[Dict[str, float]] = None,
# ):
#     """
#     Adds an inset with group mean±SEM JNDs for each view.
#     Uses small x offsets so multiple genotypes at same ABL don't overlap.
#     """

#     ax_inset = ax_parent.inset_axes(list(inset_rect))

#     if x_offsets is None:
#         # reasonable defaults for up to 3 views
#         offs = [-0.25, 0.0, 0.25]
#         x_offsets = {vn: offs[i % len(offs)] for i, vn in enumerate(view_names)}

#     # Collect all abls present
#     all_abls = sorted(set().union(*[
#         set(group_jnd_by_view.get(vn, pd.DataFrame()).get("ABL", []))
#         for vn in view_names
#     ]))

#     for vn in view_names:
#         dfj = group_jnd_by_view.get(vn, pd.DataFrame())
#         if dfj is None or dfj.empty:
#             continue
#         c = view_colors.get(vn, "gray")
#         off = x_offsets.get(vn, 0.0)

#         for _, row in dfj.iterrows():
#             ax_inset.errorbar(
#                 float(row["ABL"]) + off, float(row["mean"]),
#                 yerr=float(row["sem"]),
#                 fmt="o",
#                 color=c,
#                 markersize=7,
#                 elinewidth=1.5,
#                 capsize=4,
#                 markeredgecolor="black",
#                 markeredgewidth=1,
#             )

#     style_axes(ax_inset, style, title=None, xlabel="ABL", ylabel="JND (dB)")
#     ax_inset.set_xticks(all_abls)
#     ax_inset.tick_params(axis="both", labelsize=max(8, style.tick_fs - 6))
#     ax_inset.set_box_aspect(1)
#     ax_inset.spines["top"].set_visible(False)
#     ax_inset.spines["right"].set_visible(False)
#     ax_inset.grid(False)

#     return ax_inset


def add_jnd_inset_abl_colored(
    ax_parent,
    group_jnd_df,
    abl_colors: dict,
    style: PlotStyle,
    inset_rect=(0.70, 0.15, 0.30, 0.30),
):
    """
    Inset for views_3x3: shows group mean±SEM JND for all ABLs,
    dot color matches the ABL curve colors.
    """
    import pandas as pd

    ax_inset = ax_parent.inset_axes(list(inset_rect))

    if group_jnd_df is None or getattr(group_jnd_df, "empty", True):
        style_axes(ax_inset, style, title=None, xlabel="ABL", ylabel="JND (dB)")
        ax_inset.set_xticks([])
        ax_inset.set_box_aspect(1)
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)
        return ax_inset

    for _, row in group_jnd_df.iterrows():
        abl = int(row["ABL"])
        c = abl_colors.get(abl, "gray")
        ax_inset.errorbar(
            abl, float(row["mean"]),
            yerr=float(row["sem"]),
            fmt="o",
            color=c,
            markersize=7,
            elinewidth=1.5,
            capsize=4,
            markeredgecolor="black",
            markeredgewidth=1,
        )

    abls = sorted(set(int(x) for x in group_jnd_df["ABL"].unique()))
    style_axes(ax_inset, style, title=None, xlabel="ABL", ylabel="JND (dB)")
    ax_inset.set_xticks(abls)
    ax_inset.tick_params(axis="both", labelsize=max(8, style.tick_fs - 6))
    ax_inset.set_box_aspect(1)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)
    ax_inset.grid(False)
    return ax_inset


def add_jnd_inset_single_abl(
    ax_parent,
    group_jnd_by_view: dict,
    view_names,
    view_colors: dict,
    abl: int,
    style: PlotStyle,
    inset_rect=(0.70, 0.15, 0.30, 0.30),
):
    """
    Inset for abls_4x3: shows ONLY the JND for this row's ABL (one per genotype/view if available).
    """
    import pandas as pd

    ax_inset = ax_parent.inset_axes(list(inset_rect))

    # x offsets for multiple genotypes at the same ABL
    offs = [-0.25, 0.0, 0.25]
    x_offsets = {vn: offs[i % len(offs)] for i, vn in enumerate(view_names)}

    plotted_any = False
    for vn in view_names:
        dfj = group_jnd_by_view.get(vn)
        if dfj is None or getattr(dfj, "empty", True):
            continue

        sub = dfj[dfj["ABL"] == abl]
        if sub.empty:
            continue

        row = sub.iloc[0]
        c = view_colors.get(vn, "gray")
        x = float(abl) + float(x_offsets.get(vn, 0.0))

        ax_inset.errorbar(
            x, float(row["mean"]),
            yerr=float(row["sem"]),
            fmt="o",
            color=c,
            markersize=7,
            elinewidth=1.5,
            capsize=4,
            markeredgecolor="black",
            markeredgewidth=1,
        )
        plotted_any = True

    style_axes(ax_inset, style, title=None, xlabel="ABL", ylabel="JND (dB)")
    ax_inset.set_xticks([abl])
    ax_inset.set_xlim(abl - 1, abl + 1)
    ax_inset.tick_params(axis="both", labelsize=max(8, style.tick_fs - 6))
    ax_inset.set_box_aspect(1)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)
    ax_inset.grid(False)

    return ax_inset
