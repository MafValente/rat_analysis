# analysis/plotting/groupcomparison/style.py

import matplotlib.pyplot as plt

def _xlim_for_abl(abl, xlim_map, default=None):
    """
    xlim_map can be:
      - None (returns default)
      - a tuple (xmin, xmax) used for all ABLs
      - a dict like {"default": (xmin, xmax), 50: (xmin, xmax), 60: (...)}
      - a callable: f(abl) -> (xmin, xmax) or None
    """
    if xlim_map is None:
        return default
    if callable(xlim_map):
        out = xlim_map(abl)
        return default if out is None else out
    if isinstance(xlim_map, tuple) and len(xlim_map) == 2:
        return xlim_map
    if isinstance(xlim_map, dict):
        return xlim_map.get(abl, xlim_map.get("default", default))
    return default

import numpy as np

# plotting/GroupComparison/style.py
import matplotlib.ticker as mticker

def relabel_ticks_minus18_plus18_as_50(ax, eps=1e-6):
    """
    Relabel tick *values* near ±18 as ±50 using a formatter.
    No set_ticklabels() (so no warnings / no desync).
    """
    def _fmt(x, pos=None):
        if abs(x + 18) < eps:
            return "-50"
        if abs(x - 18) < eps:
            return "50"
        # integers look clean
        if abs(x - round(x)) < eps:
            return str(int(round(x)))
        return f"{x:g}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))

def style_axes(ax, title, xlabel, ylabel,
              TITLE_FONTSIZE=24, LABEL_FONTSIZE=25, TICK_FONTSIZE=24, TITLE_PAD=16):
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color="black")

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE,
                   colors="black", width=1.5, length=6)

    for s in ["left", "right", "top", "bottom"]:
        if s in ax.spines:
            ax.spines[s].set_color("black")
            ax.spines[s].set_linewidth(1.5)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

        
# def relabel_ticks_minus18_plus18_as_50(ax, eps=1e-6):
#     xticks = list(ax.get_xticks())

#     # ensure -18 and +18 are present
#     if not any(abs(x + 18) < eps for x in xticks):
#         xticks.append(-18.0)
#     if not any(abs(x - 18) < eps for x in xticks):
#         xticks.append(18.0)

#     xticks = sorted(xticks)
#     ax.set_xticks(xticks)

#     labels = []
#     for x in xticks:
#         if abs(x + 18) < eps:
#             labels.append("-50")
#         elif abs(x - 18) < eps:
#             labels.append("50")
#         else:
#             # keep integers clean, otherwise show the value
#             labels.append(str(int(round(x))) if abs(x - round(x)) < 1e-6 else f"{x:g}")

#     ax.set_xticklabels(labels)
