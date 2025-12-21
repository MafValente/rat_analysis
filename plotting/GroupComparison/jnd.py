# analysis/plotting/groupcomparison/jnd.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Psychometric
import Helpers.DataHelpers as DataHelpers


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else x.std(ddof=1) / np.sqrt(len(x))


def compute_group_jnd(df_view, skip_ABL=50):
    all_jnds = []

    for subject, df_subj in df_view.groupby("animal"):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=skip_ABL)
        if jnd_df.empty:
            continue
        jnd_df["subject"] = subject
        all_jnds.append(jnd_df)

    if not all_jnds:
        return pd.DataFrame(columns=["ABL", "mean", "sem", "n"]), pd.DataFrame()

    all_jnds_df = pd.concat(all_jnds, ignore_index=True)

    group_jnd = (
        all_jnds_df.groupby("ABL")["JND"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    return group_jnd, all_jnds_df


def plot_old_vs_new_jnd_scatter(old_jnd_data, all_jnds_df,
                               LABEL_FONTSIZE=25, TICK_FONTSIZE=24):
    """
    Returns a matplotlib Figure. Does NOT call plt.show().
    """
    import matplotlib.pyplot as plt

    old_ABLs = old_jnd_data["ABLS"]
    old_jnds = old_jnd_data["jnds"]
    old_animals = old_jnd_data["animals_with_mean"]

    ABL_COLOR_MAP = {20: "C0", 40: "C1", 60: "C3"}

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # OLD (open circles)
    for abl in old_ABLs:
        color = ABL_COLOR_MAP.get(abl, "gray")
        if abl not in old_jnds:
            continue
        for animal in old_animals:
            if animal in old_jnds[abl]:
                ax.scatter(
                    abl - 0.5,
                    old_jnds[abl][animal],
                    facecolors="none",
                    edgecolors=color,
                    s=60, lw=1, alpha=0.9,
                )

    # NEW (closed circles)
    if all_jnds_df is not None and len(all_jnds_df) > 0:
        for animal, df_an in all_jnds_df.groupby("subject"):
            for abl in sorted(df_an["ABL"].unique()):
                color = ABL_COLOR_MAP.get(abl, "gray")
                jnd_val = df_an.loc[df_an["ABL"] == abl, "JND"].values[0]
                ax.scatter(
                    abl + 0.5,
                    jnd_val,
                    color=color,
                    s=55, alpha=0.7,
                    edgecolor="black", linewidth=0.5,
                )

    ax.set_xlabel("ABL (dB)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("JND (dB)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(sorted(ABL_COLOR_MAP.keys()))
    ax.set_xlim(15, 65)

    plt.tight_layout()
    return fig
