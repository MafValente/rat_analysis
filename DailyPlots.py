#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Psychometric 
import Helpers.DataHelpers as DataHelpers
import matplotlib as mpl
import pickle

subject_file = "merged_ASD0022.csv"

# ==============================================================
# CONFIG: choose which line you're analyzing
# ==============================================================
LINE = "CNTNAP2"   # or "SHANK3"
COHORT = "cohort2" # or "cohort1", etc

BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"

LINE_ROOTS = {
     ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
     ("SHANK3", "cohort1"): "SHANK3_cohort1",
 }

DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[LINE,COHORT])

os.chdir(DATA_DIR)

df = pd.read_csv(subject_file)

"""



 .d888b,?88   d8P  88bd8b,d88b   88bd8b,d88b  d888b8b    88bd88b?88   d8P
 ?8b,   d88   88   88P'`?8P'?8b  88P'`?8P'?8bd8P' ?88    88P'  `d88   88
   `?8b ?8(  d88  d88  d88  88P d88  d88  88P88b  ,88b  d88     ?8(  d88
`?888P' `?88P'?8bd88' d88'  88bd88' d88'  88b`?88P'`88bd88'     `?88P'?8b
                                                                       )88
                                                                      ,d8P
                                                                   `?888P'
           d8b
           88P            d8P
          d88          d888888P
?88,.d88b,888   d8888b   ?88'   .d888b,
`?88'  ?88?88  d8P' ?88  88P    ?8b,
  88b  d8P 88b 88b  d88  88b      `?8b
  888888P'  88b`?8888P'  `?8b  `?888P'
  88P'
 d88
 ?8P

                                                    d8P
                                                 d888888P
?88,.d88b, d8888b  88bd88b      88bd88b d888b8b    ?88'
`?88'  ?88d8b_,dP  88P'  `      88P'  `d8P' ?88    88P
  88b  d8P88b     d88          d88     88b  ,88b   88b
  888888P'`?888P'd88'         d88'     `?88P'`88b  `?8b
  88P'
 d88
 ?8P
"""

# ==============================================================
# === CONFIG (Aesthetic & Behavior) =============================
# ==============================================================
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 22
TICK_FONTSIZE  = 20
LEGEND_FONTSIZE = 16
TITLE_PAD = 16
LINEWIDTH = 1.8

mpl.rcParams.update({
    "savefig.pad_inches": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})
colors = ["C0", "C1", "C2", "C3"]
# ==============================================================
# === HELPER FUNCTIONS =========================================
# ==============================================================
def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to an axis."""
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color="black")

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE,
                   colors="black", width=1.5, length=6)
    for s in ["left", "bottom", "right", "top"]:
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_box_aspect(1)

def load_makefig1_data(pkl_path="fig1_plot_data.pkl"):
#Load the pickled psychometric dataset from make_fig1.
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_makefig1_chrono(pkl_path="fig1_chrono_plot_data.pkl"):
    """Load the chronometric (RT) data from make_fig1."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


# ==============================================================
# === Load reference (neurotypical) datasets ===================
# ==============================================================

makefig1_data = load_makefig1_data(
    "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_plot_data.pkl"
)
makefig1_chrono = load_makefig1_chrono(
    "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_chrono_plot_data.pkl"
)

makefig1_data = DataHelpers.normalize_ABL_labels(makefig1_data)

df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"] == False].copy()

df = df[df["training_level"]==16].copy()

# Prep session data
df_last = df
#df_last, last_sessions = DataHelpers.get_last_n_sessions(df, 33)


n_trials = len(df_last)
session_type = df_last["session_type"].unique()
session_number = df_last["session"].nunique()
setup_number = df_last["box"].unique()
animal_number = df_last["animal"].iloc[0]

# ==============================================================
# === COMPUTE SUMMARY STATS =====================================
# ==============================================================
meanRT = (
    df_last[df_last["success"] == 1]
    .groupby(["ABL", "ILD"])["timed_rt"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

meanMT = (
    df_last[df_last["success"] == 1]
    .groupby(["ABL", "ILD"])["timed_mt"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

results = Psychometric.compute_psychometrics_by_ABL(df_last)


# ==============================================================
# === PLOTTING ==================================================
# ==============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Info text (top-left) ---
ax_text = axes[0,0]
ax_text.axis("off")
info = (
    f"Animal: {animal_number}\n"
    f"Setup number: {setup_number}\n"
    f"Session type: {session_type}\n"
    f"Sessions: {session_number}\n"
    f"Number of trials: {n_trials}\n"
)
ax_text.text(
    0.5, 0.5, info,
    fontsize=16,
    ha="center", va="center",
    family="monospace"
)


# --- RT panel (top-right) ---
for i, abl in enumerate(sorted(meanRT["ABL"].unique())):
    sub = meanRT[meanRT["ABL"] == abl]
    axes[0,1].errorbar(
        DataHelpers.shift_ILD_for_ABL50(sub["ILD"]), sub["mean"],
        yerr=sub["std"] / np.sqrt(sub["count"]),
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="none", marker="o", capsize=5, linewidth=LINEWIDTH
    )

"""
# --- Individual neurotypical RTs (colored by ABL) ---
DataHelpers.overlay_makefig1_rt_individuals(
    axes[0,1],
    makefig1_chrono,
    abl=None,                # None → all ABLs, or specify 40 for just that one
    show_individuals=True,
    use_abl_colors=True,     # True = color by ABL; False = gray
    alpha=0.35,
)


# --- Overlay neurotypical mean RT ---
DataHelpers.overlay_makefig1_rt(axes[0,1], abl, makefig1_chrono, color="black", zorder=1)
"""


_style_axes(
    axes[0,1],
    title="Reaction Time",
    xlabel="ILD (dB)",
    ylabel="Mean Reaction Time (s)"
)


# --- MT panel (bottom-left) ---
for i, abl in enumerate(sorted(meanMT["ABL"].unique())):
    sub = meanMT[meanMT["ABL"] == abl]
    axes[1,0].errorbar(
        DataHelpers.shift_ILD_for_ABL50(sub["ILD"]), sub["mean"],
        yerr=sub["std"] / np.sqrt(sub["count"]),
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="none", marker="o", capsize=5, linewidth=LINEWIDTH
    )

_style_axes(
    axes[1,0],
    title="Movement Time",
    xlabel="ILD (dB)",
    ylabel="Mean Movement Time (s)"
)

# --- Psychometric (bottom-right) ---
ax_psy = axes[1,1]

for color, (abl, res) in zip(colors, results.items()):
    ax_psy.scatter(
        DataHelpers.shift_ILD_for_ABL50(res["ILDs"]),
        res["PropLeft"],
        label=f"ABL={abl}",
        color=color,
        s=70,
        edgecolor=color,
        linewidth=0.6,
        zorder=3,
    )
    if abl != 50:
        ax_psy.plot(
            DataHelpers.shift_ILD_for_ABL50(res["xx"]),
            res["yy"],
            color=color,
            linewidth=LINEWIDTH
        )


# Overlay both mean and individual neurotypical psychometrics
DataHelpers.overlay_makefig1_psychometrics(axes[1,1], makefig1_data, color="black", show_individuals=True, use_abl_colors=False)



# Dashed lines for 0 and 50%
plt.axvline(0, color='black', linestyle='--', linewidth=0.8, zorder=-100)
plt.axhline(0.5, color='black', linestyle='--', linewidth=0.8, zorder=-100)

# --- Axis styling ---
_style_axes(
    ax_psy,
    title="Psychometric by ABL",
    xlabel="ILD (dB)",
    ylabel="Proportion Left"
)

# --- Shared tick adjustments for ±50 mapping ---
xticks = sorted(set(ax_psy.get_xticks()) | {-18, 18})
ax_psy.set_xticks(xticks)
ax_psy.set_xticklabels([
    "-50" if x == -18 else
    "50" if x == 18 else
    str(int(x))
    for x in xticks
])
ax_psy.set_xlim(-19, 19)

ax_psy.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

plt.tight_layout()
plt.show()

#%%

# --- Compute JNDs using DataHelpers ---
jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=50)

# --- Plot ---

fig_jnd, ax_jnd = plt.subplots(figsize=(2.5, 2))  # small: 3 inches wide × 2.5 high
ax_jnd.plot(jnd_df["ABL"], jnd_df["JND"], "-o", color="black",
            linewidth=1.2, markersize=4)

_style_axes(ax_jnd,
    xlabel="ABL (dB)",
    ylabel="JND (dB)"
)

ax_jnd.set_xticks(sorted(jnd_df["ABL"].unique()))
ax_jnd.grid(True, linestyle="--", alpha=0.3, zorder=-10)


ax_jnd.set_xlabel("ABL", fontsize=10)
ax_jnd.set_ylabel("JND", fontsize=10)
ax_jnd.tick_params(labelsize=10)
plt.tight_layout()
plt.show()


#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Psychometric 
import Helpers.DataHelpers as DataHelpers
import matplotlib as mpl

"""

                                                                     d8P
                                                                  d888888P
 d888b8b   d888b8b   d888b8b    88bd88b d8888b d888b8b   d888b8b    ?88'   d8888b
d8P' ?88  d8P' ?88  d8P' ?88    88P'  `d8b_,dPd8P' ?88  d8P' ?88    88P   d8b_,dP
88b  ,88b 88b  ,88b 88b  ,88b  d88     88b    88b  ,88b 88b  ,88b   88b   88b
`?88P'`88b`?88P'`88b`?88P'`88bd88'     `?888P'`?88P'`88b`?88P'`88b  `?8b  `?888P'
                 )88       )88                       )88
                ,88P      ,88P                      ,88P
            `?8888P   `?8888P                   `?8888P



 .d888b,?88   d8P  88bd8b,d88b   88bd8b,d88b  d888b8b    88bd88b?88   d8P
 ?8b,   d88   88   88P'`?8P'?8b  88P'`?8P'?8bd8P' ?88    88P'  `d88   88
   `?8b ?8(  d88  d88  d88  88P d88  d88  88P88b  ,88b  d88     ?8(  d88
`?888P' `?88P'?8bd88' d88'  88bd88' d88'  88b`?88P'`88bd88'     `?88P'?8b
                                                                       )88
                                                                      ,d8P
                                                                   `?888P'
"""

"""
=============================================================
 Group Summary — with GroupComparison Aesthetics
=============================================================
"""

# ==============================================================
# CONFIG: choose which line you're analyzing
# ==============================================================
LINE = "CNTNAP2"   # or "SHANK3"
COHORT = "cohort2" # or "cohort1", etc

BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"

LINE_ROOTS = {
     ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
     ("SHANK3", "cohort1"): "SHANK3_cohort1",
 }

DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[LINE, COHORT])

os.chdir(DATA_DIR)

# ==============================================================
# === LOAD & FILTER DATA =======================================
# ==============================================================

cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file)

df = df[df["training_level"] == 16]
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"]==False].copy()

# ==============================================================
# === CONFIG (Aesthetic constants) =============================
# ==============================================================
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 22
TICK_FONTSIZE  = 20
LEGEND_FONTSIZE = 16
TITLE_PAD = 16
LINEWIDTH = 2.0

mpl.rcParams.update({
    "savefig.pad_inches": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

colors = ["C0", "C1", "C2", "C3"]
SKIP_PSY_FITS = {50}


# ==============================================================
# === HELPERS ==================================================
# ==============================================================
def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent formatting."""
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color="black")

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE,
                   colors="black", width=1.5, length=6)
    for s in ["left", "bottom", "right", "top"]:
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)


def sem(series):
    n = series.count()
    return series.std(ddof=1) / np.sqrt(n) if n > 0 else np.nan


# ==============================================================
# === CONTRIBUTOR STATS ========================================
# ==============================================================
contributors_overall = sorted(df[df["success"] == 1]["animal"].unique())
print(f"Overall contributors ({len(contributors_overall)}):", contributors_overall)

contributors_by_abl = (
    df[df["success"] != 0]
    .groupby("ABL")["animal"]
    .agg(lambda s: sorted(s.unique()))
    .to_dict()
)
print("\nContributors by ABL:")
for abl, subs in contributors_by_abl.items():
    print(f"  ABL {abl} (n={len(subs)}): {subs}")

# By box 
contributors_by_setup = (
    df[df["success"] != 0]
    .groupby(["box"])["animal"]
    .agg(subjects=lambda s: sorted(s.unique()),
         n_subjects=lambda s: s.nunique())
    .reset_index()
)

# ==============================================================
# === COMPUTE GROUP METRICS ====================================
# ==============================================================

# RT
meanRT_per_subject = (
    df[df["success"] == 1]
    .groupby(["animal", "ABL", "ILD"])["timed_rt"]
    .mean()
    .reset_index(name="mean_rt")
)

meanRT_grouped = (
    meanRT_per_subject
    .groupby(["ABL", "ILD"])["mean_rt"]
    .agg(mean="mean", sem=sem)
    .reset_index()
)

# MT
meanMT_per_subject = (
    df[df["success"] == 1]
    .groupby(["animal", "ABL", "ILD"])["timed_mt"]
    .mean()
    .reset_index(name="mean_mt")
)

meanMT_grouped = (
    meanMT_per_subject
    .groupby(["ABL", "ILD"])["mean_mt"]
    .agg(mean="mean", sem=sem)
    .reset_index()
)

# Psychometric
all_subject_points = []
for subject, df_subj in df.groupby("animal"):
    results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
    for abl, res in results.items():
        for ild, pleft in zip(res["ILDs"], res["PropLeft"]):
            all_subject_points.append({
                "subject": subject,
                "ABL": abl,
                "ILD": ild,
                "PropLeft": pleft
            })

points_df = pd.DataFrame(all_subject_points)
agg_points = (
    points_df.groupby(["ABL", "ILD"])["PropLeft"]
    .agg(mean="mean", sem=sem)
    .reset_index()
)


# Fit per ABL
group_fits = {}
for abl in sorted(agg_points["ABL"].unique()):
    sub = agg_points[agg_points["ABL"] == abl]
    ILDs = DataHelpers.shift_ILD_for_ABL50(sub["ILD"])
    PropLeft_mean = sub["mean"].values
    n_trials = np.full_like(ILDs, 50, dtype=float)
    pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
        ILDs, PropLeft_mean, model="my_psycho", n_trials=n_trials, show_plot=False
    )
    xx = np.linspace(min(ILDs), max(ILDs), 200)
    group_fits[abl] = {
        "ILDs": ILDs,
        "mean": PropLeft_mean,
        "sem": sub["sem"].values,
        "xx": xx,
        "yy": yy
    }


# ==============================================================
# === Compute JND per subject per ABL ===========================
# ==============================================================

all_jnds = []

for subject, df_subj in df.groupby("animal"):
    results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")

    # Compute JNDs for this subject (using your DataHelpers helper)
    jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=50)

    # Store subject ID
    jnd_df["subject"] = subject
    all_jnds.append(jnd_df)

# Combine all subjects
all_jnds_df = pd.concat(all_jnds, ignore_index=True)

# Compute group mean ± SEM
group_jnd = (
    all_jnds_df.groupby("ABL")["JND"]
    .agg(mean="mean", sem=sem, n="count")
    .reset_index()
)
print(group_jnd)


# ==============================================================
# === PLOTTING ==================================================
# ==============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- RT ---
for i, abl in enumerate(sorted(meanRT_grouped["ABL"].unique())):
    sub = meanRT_grouped[meanRT_grouped["ABL"] == abl]
    axes[0].errorbar(
        DataHelpers.shift_ILD_for_ABL50(sub["ILD"]), sub["mean"],
        yerr=sub["sem"], color=colors[i % len(colors)],
        linestyle="none", marker="o", capsize=5, linewidth=LINEWIDTH,
        label=f"ABL {abl}"
    )
_style_axes(axes[0], "Mean RT", "ILD (dB)", "Mean RT (s)")
axes[0].legend(fontsize=LEGEND_FONTSIZE, frameon=False)

# --- MT ---
for i, abl in enumerate(sorted(meanMT_grouped["ABL"].unique())):
    sub = meanMT_grouped[meanMT_grouped["ABL"] == abl]
    axes[1].errorbar(
        DataHelpers.shift_ILD_for_ABL50(sub["ILD"]), sub["mean"],
        yerr=sub["sem"], color=colors[i % len(colors)],
        linestyle="none", marker="o", capsize=5, linewidth=LINEWIDTH,
        label=f"ABL {abl}"
    )
_style_axes(axes[1], "Mean MT", "ILD (dB)", "Mean MT (s)")

# --- Psychometric ---
for i, (abl, res) in enumerate(group_fits.items()):
    axes[2].errorbar(
        res["ILDs"], res["mean"], yerr=res["sem"],
        fmt="o", color=colors[i % len(colors)],
        capsize=3, label=f"ABL {abl}"
    )
    if abl not in SKIP_PSY_FITS and "xx" in res and "yy" in res:
        axes[2].plot(res["xx"], res["yy"], color=colors[i], linewidth=LINEWIDTH)
_style_axes(axes[2], "Psychometric by ABL", "ILD (dB)", "Proportion Left")
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=LEGEND_FONTSIZE, frameon=False)

# --- Tick relabeling for ±18 ↔ ±50 ---
for ax in axes.flatten():
    xticks = sorted(set(ax.get_xticks()) | {-18, 18})
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        "-50" if x == -18 else
        "50" if x == 18 else
        str(int(x))
        for x in xticks
    ])
    ax.set_xlim(-19, 19)

plt.tight_layout()
plt.show()

#%%
# ==============================================================
# === Plot GROUP JNDs ==========================================
# ==============================================================

fig_jnd, ax_jnd = plt.subplots(figsize=(3, 2.5))

ax_jnd.errorbar(
    group_jnd["ABL"], group_jnd["mean"],
    yerr=group_jnd["sem"],
    fmt="o-", color="black", capsize=4, linewidth=LINEWIDTH
)

_style_axes(ax_jnd,
            title="Group JND",
            xlabel="ABL (dB)",
            ylabel="JND (ILD units)")

ax_jnd.set_xticks(sorted(group_jnd["ABL"].unique()))
ax_jnd.tick_params(axis="both", labelsize=TICK_FONTSIZE - 4)
ax_jnd.set_box_aspect(1)
plt.tight_layout()
plt.show()

#%%

"""

                                                d8P
                                             d888888P
?88,.d88b, d8888b  88bd88b     .d888b, d8888b  ?88'  ?88   d8P?88,.d88b,
`?88'  ?88d8b_,dP  88P'  `     ?8b,   d8b_,dP  88P   d88   88 `?88'  ?88
  88b  d8P88b     d88            `?8b 88b      88b   ?8(  d88   88b  d8P
  888888P'`?888P'd88'         `?888P' `?888P'  `?8b  `?88P'?8b  888888P'
  88P'                                                          88P'
 d88                                                           d88
 ?8P                                                           ?8P
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import Psychometric
import Helpers.DataHelpers as DataHelpers

"""
=============================================================
 Group-by-Setup Summary — with GroupComparison Aesthetics
=============================================================
"""
# ==============================================================
# CONFIG: choose which line you're analyzing
# ==============================================================
LINE = "CNTNAP2"   # or "SHANK3"
COHORT = "cohort2" # or "cohort1", etc

BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"

LINE_ROOTS = {
     ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
     ("SHANK3", "cohort1"): "SHANK3_cohort1",
 }

DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[LINE,COHORT])

os.chdir(DATA_DIR)

# ==============================================================
# === CONFIG ===================================================
# ==============================================================
SETUP_COL = "box"
PALETTE = "Accent"
SKIP_PSY_FITS = {50}

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE  = 20
LEGEND_FONTSIZE = 16
TITLE_PAD = 16
LINEWIDTH = 2.0

mpl.rcParams.update({
    "savefig.pad_inches": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ==============================================================
# === LOAD & FILTER DATA =======================================
# ==============================================================
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file)
df = df[df["session"] >= 13]

df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"]==False].copy()

if SETUP_COL not in df.columns:
    raise KeyError(f"'{SETUP_COL}' column not found in dataframe.")

ABLs = sorted(df["ABL"].unique())
setups = sorted(df[SETUP_COL].dropna().unique())

# Color map per setup
cmap = plt.cm.get_cmap(PALETTE)
color_for_setup = {s: cmap(i / max(len(setups) - 1, 1)) for i, s in enumerate(setups)}

# ==============================================================
# === STYLE HELPERS ============================================
# ==============================================================
def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent formatting."""
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color="black")

    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE,
                   colors="black", width=1.5, length=6)
    for s in ["left", "bottom", "right", "top"]:
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)



def sem(series):
    n = series.count()
    return series.std(ddof=1) / np.sqrt(n) if n > 0 else np.nan

# ==============================================================
# === COMPUTE STATS ============================================
# ==============================================================
# --- RT ---
meanRT_per_subject = (
    df[df["success"] == 1]
    .groupby([SETUP_COL, "animal", "ABL", "ILD"])["timed_rt"]
    .mean()
    .reset_index(name="mean_rt")
)
meanRT_setup = (
    meanRT_per_subject
    .groupby([SETUP_COL, "ABL", "ILD"])["mean_rt"]
    .agg(mean="mean", sem=sem)
    .reset_index()
)
rt_counts = meanRT_per_subject.groupby([SETUP_COL, "ABL"])["animal"].nunique().to_dict()


# --- MT ---
meanMT_per_subject = (
    df[df["success"] == 1]
    .groupby([SETUP_COL, "animal", "ABL", "ILD"])["timed_mt"]
    .mean()
    .reset_index(name="mean_mt")
)
meanMT_setup = (
    meanMT_per_subject
    .groupby([SETUP_COL, "ABL", "ILD"])["mean_mt"]
    .agg(mean="mean", sem=sem)
    .reset_index()
)
mt_counts = meanMT_per_subject.groupby([SETUP_COL, "ABL"])["animal"].nunique().to_dict()

# --- Psychometric ---
all_points = []
for (setup, subject), df_ss in df.groupby([SETUP_COL, "animal"]):
    results = Psychometric.compute_psychometrics_by_ABL(df_ss, model="erf_psycho")
    for abl, res in results.items():
        for ild, pleft in zip(res["ILDs"], res["PropLeft"]):
            all_points.append({
                SETUP_COL: setup,
                "animal": subject,
                "ABL": abl,
                "ILD": ild,
                "PropLeft": pleft
            })

points_df = pd.DataFrame(all_points)
agg_points_setup = (
    points_df
    .groupby([SETUP_COL, "ABL", "ILD"])["PropLeft"]
    .agg(mean="mean", sem=sem)
    .reset_index()
)
psy_counts = points_df.groupby([SETUP_COL, "ABL"])["animal"].nunique().to_dict()


# --- Fit curves per ABL × setup ---
group_fits = {abl: {} for abl in ABLs}
for abl in ABLs:
    for setup in setups:
        sub = agg_points_setup[(agg_points_setup["ABL"] == abl) & (agg_points_setup[SETUP_COL] == setup)]
        if sub.empty:
            continue
        ILDs =sub["ILD"]
        PropLeft_mean = sub["mean"].values
        n_trials = np.full_like(ILDs, 50, dtype=float)
        pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
            ILDs, PropLeft_mean, model="erf_psycho", n_trials=n_trials, show_plot=False
        )
        xx = np.linspace(np.min(ILDs), np.max(ILDs), 200)
        group_fits[abl][setup] = {
            "ILDs": ILDs, "mean": PropLeft_mean, "sem": sub["sem"].values,
            "xx": xx, "yy": yy
        }
# ==============================================================
# === PLOTTING ==================================================
# ==============================================================
# Keep each panel compact: ~5.5×3.2 inches per panel
panel_w, panel_h = 5.5, 5.2
fig_height = panel_h * len(ABLs)
fig_width = panel_w * 3
fig, axes = plt.subplots(len(ABLs), 3, figsize=(fig_width, fig_height), sharex="col")

if len(ABLs) == 1:
    axes = np.array([axes])  # ensure 2D for single row

for r, abl in enumerate(ABLs):
    # --- RT ---
    ax = axes[r, 0]
    for setup in setups:
        sub = meanRT_setup[(meanRT_setup["ABL"] == abl) & (meanRT_setup[SETUP_COL] == setup)]
        if sub.empty:
            continue
        n_subs = rt_counts.get((setup, abl), 0)
        sub = sub.sort_values("ILD")
        ax.errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]), sub["mean"], yerr=sub["sem"],
            label=f"{setup} (n={n_subs})",
            color=color_for_setup[setup], linestyle="none",
            marker="o", capsize=4, linewidth=LINEWIDTH
        )
    _style_axes(ax, f"ABL {abl} — RT", "ILD (dB)", "Mean RT (s)")

    # --- MT ---
    ax = axes[r, 1]
    for setup in setups:
        sub = meanMT_setup[(meanMT_setup["ABL"] == abl) & (meanMT_setup[SETUP_COL] == setup)]
        if sub.empty:
            continue
        n_subs = mt_counts.get((setup, abl), 0)
        sub = sub.sort_values("ILD")
        ax.errorbar(
            DataHelpers.shift_ILD_for_ABL50(sub["ILD"]), sub["mean"], yerr=sub["sem"],
            label=f"{setup} (n={n_subs})",
            color=color_for_setup[setup], linestyle="none",
            marker="o", capsize=4, linewidth=LINEWIDTH
        )
    _style_axes(ax, None, "ILD (dB)" if r == len(ABLs) - 1 else None, "Mean MT (s)")


    # --- Psychometric ---
    ax = axes[r, 2]
    for setup in setups:
        res = group_fits.get(abl, {}).get(setup, None)
        if res is None:
            continue
        n_subs = psy_counts.get((setup, abl), 0)
        order = np.argsort(res["ILDs"])
        ax.errorbar(
            np.array(res["ILDs"])[order], np.array(res["mean"])[order],
            yerr=np.array(res["sem"])[order],
            fmt="o", color=color_for_setup[setup], capsize=3,
            label=f"{setup} (n={n_subs})" if r == 0 else None
        )
        if abl not in SKIP_PSY_FITS and "xx" in res and "yy" in res:
            ax.plot(res["xx"], res["yy"], color=color_for_setup[setup], linewidth=LINEWIDTH)
    ax.set_ylim(0, 1)
    _style_axes(ax, None, "ILD (dB)" if r == len(ABLs) - 1 else None, "P(Left)")

# --- Shared formatting ---
for ax in axes.flatten():
    xticks = sorted(set(ax.get_xticks()) | {-18, 18})
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        "-50" if x == -18 else "50" if x == 18 else str(int(x))
        for x in xticks
    ])
    ax.set_xlim(-19, 19)

# --- Legend only for ABL = 50 (on RT panel) ---
for r, abl in enumerate(ABLs):
    if abl == 50:
        axes[r, 0].legend(fontsize=LEGEND_FONTSIZE, ncol=1, frameon=False)
    else:
        if axes[r, 0].get_legend(): axes[r, 0].get_legend().remove()
    if axes[r, 1].get_legend(): axes[r, 1].get_legend().remove()
    if axes[r, 2].get_legend(): axes[r, 2].get_legend().remove()


# Adjust spacing between subplot rows and columns
plt.subplots_adjust(
    hspace=0.45,  # vertical space between rows (↑ this to increase separation)
    wspace=0.25   # horizontal spacing between columns
)
plt.show()


plt.show()


# %%
