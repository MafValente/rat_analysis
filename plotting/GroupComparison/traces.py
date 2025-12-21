# analysis/plotting/groupcomparison/traces.py

import Helpers.DataHelpers as DataHelpers
import matplotlib.pyplot as plt

# at top of traces.py
def _xcol(df):
    return "abs_ILD" if "abs_ILD" in df.columns else "ILD"

def _ild_series(df):
    """Return the column to use for x-axis (chronometric): prefer 'ILD', fallback to 'abs_ILD'."""
    if "ILD" in df.columns:
        return df["ILD"]
    if "abs_ILD" in df.columns:
        return df["abs_ILD"]
    raise KeyError("Expected column 'ILD' (or 'abs_ILD') in prepared tables.")


def plot_rt(ax, abl, tables, color, mode):
    rt_group = tables["rt_group"]
    rt_per_subj = tables["rt_per_subj"]

    if mode == "sem":
        xcol = _xcol(sub)
        sub = sub.sort_values(xcol)
        if sub.empty:
            return
        x = DataHelpers.shift_ILD_for_ABL50(sub[xcol])
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                    markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
        ax.plot(x, sub["mean"], color=color, linewidth=2.0)
        return

    # individuals mode
    for _, df_an in rt_per_subj[rt_per_subj["ABL"] == abl].groupby("animal"):
        df_an = df_an.sort_values("ILD")
        ax.plot(DataHelpers.shift_ILD_for_ABL50(_ild_series(df_an)), df_an["mean_rt"],
                color=color, alpha=0.35, linewidth=1.5)

    sub = rt_group[rt_group["ABL"] == abl].sort_values("ILD")
    if sub.empty:
        return
    x = DataHelpers.shift_ILD_for_ABL50(_ild_series(sub))
    ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3, zorder=3)
    ax.plot(x, sub["mean"], color=color, linewidth=2.5, zorder=2)


def plot_mt(ax, abl, tables, color, mode):
    mt_group = tables["mt_group"]
    mt_per_subj = tables["mt_per_subj"]

    if mode == "sem":
        xcol = _xcol(sub)
        sub = sub.sort_values(xcol)
        if sub.empty:
            return
        x = DataHelpers.shift_ILD_for_ABL50(sub[xcol])
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                    markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
        ax.plot(x, sub["mean"], color=color, linewidth=2.0)
        return

    for _, df_an in mt_per_subj[mt_per_subj["ABL"] == abl].groupby("animal"):
        df_an = df_an.sort_values("ILD")
        ax.plot(DataHelpers.shift_ILD_for_ABL50(_ild_series(df_an)), df_an["mean_mt"],
                color=color, alpha=0.35, linewidth=1.5)

    sub = mt_group[mt_group["ABL"] == abl].sort_values("ILD")
    if sub.empty:
        return
    x = DataHelpers.shift_ILD_for_ABL50(_ild_series(sub))
    ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3, zorder=3)
    ax.plot(x, sub["mean"], color=color, linewidth=2.5, zorder=2)


def plot_psy(ax, abl, tables, color, mode, skip_fit=None):
    psy_group = tables["psy_group"]
    psy_indiv = tables["psy_indiv_curves"]
    psy_mean = tables["psy_mean_fits"]

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1)

    # points
    sub = psy_group[psy_group["ABL"] == abl].sort_values("ILD")
    if not sub.empty:
        x = DataHelpers.shift_ILD_for_ABL50(_ild_series(sub))
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                    markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)

    # fits
    if skip_fit and abl in skip_fit:
        return

    if mode == "individuals":
        for (subject, abl_key), curve in psy_indiv.items():
            if abl_key != abl:
                continue
            ax.plot(DataHelpers.shift_ILD_for_ABL50(curve["xx"]), curve["yy"],
                    color=color, alpha=0.3, linewidth=1)

    mean_fit = psy_mean.get(abl)
    if mean_fit:
        ax.plot(DataHelpers.shift_ILD_for_ABL50(mean_fit["xx"]), mean_fit["yy"],
                color=color, linewidth=3)