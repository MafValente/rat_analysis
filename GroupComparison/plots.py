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


# ---------------------------
# JND plotting
# ---------------------------

def _default_abl_color_map_from_abls(abls: Sequence[int]) -> Dict[int, str]:
    # match your usual convention if possible
    preferred = {20: "C0", 40: "C1", 60: "C3"}
    out = {}
    i = 0
    for abl in sorted(set(int(a) for a in abls)):
        if abl in preferred:
            out[abl] = preferred[abl]
        else:
            out[abl] = f"C{i % 10}"
            i += 1
    return out


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
