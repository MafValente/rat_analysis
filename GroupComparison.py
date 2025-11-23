#%%

"""
..######..####.##....##..######...##.......########....########...#######..##......##
.##....##..##..###...##.##....##..##.......##..........##.....##.##.....##.##..##..##
.##........##..####..##.##........##.......##..........##.....##.##.....##.##..##..##
..######...##..##.##.##.##...####.##.......######......########..##.....##.##..##..##
.......##..##..##..####.##....##..##.......##..........##...##...##.....##.##..##..##
.##....##..##..##...###.##....##..##.......##..........##....##..##.....##.##..##..##
..######..####.##....##..######...########.########....##.....##..#######...###..###.
"""

"""
GroupComparison (original look, with ILD remapping)
- Only modification: plot ABL ±50 at x=±18 but label ticks as ±50.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pickle
import Psychometric
import DataHelpers
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ==============================================================
# === Load reference (neurotypical) datasets ===================
# ==============================================================
def load_makefig1_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_makefig1_chrono(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

makefig1_data = load_makefig1_data(
    "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_plot_data.pkl"
)
makefig1_chrono = load_makefig1_chrono(
    "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_chrono_plot_data.pkl"
)

# --- Remap ABL 35 → 40 ---
if 35 in makefig1_data["ABLS"]:
    makefig1_data["ABLS"] = [40 if x == 35 else x for x in makefig1_data["ABLS"]]

    # Update all relevant dicts that use ABL as key
    for key in ["ilds_dict", "mean_sigmoid_dict", "mean_params_dict", "x_smooth_dict"]:
        if 35 in makefig1_data[key]:
            makefig1_data[key][40] = makefig1_data[key].pop(35)

# ==============================================================
# === CONFIG ===================================================
# ==============================================================
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2")
cohort_file = "merged_all_subjects.csv"

error_mode = "individuals"  # "sem" or "individuals"
MASK_59_TO_60 = True
MASK_25_TO_50_WHEN_TL16 = True
TRAINING_MIN = 16
SESSION_MIN = 13

views = [
    (
        "wt",
        lambda d: DataHelpers.restrict_subjects(
            d,
            "sex_gen.csv",
            genotypes="wt",
            subject_col="animal",
            genotype_col="genotype",
            attach_meta=True,
        ),
    ),
]

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
TITLE_PAD = 16

mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "sans-serif",
]

FEMALE = "#e75480"
MALE = "#1f77b4"
colors = ["C0", "C1", "C2", "C3"]
preferred_view_colors = {"Females": FEMALE, "Males": MALE}
SKIP_PSY_FITS = {50}


# ==============================================================
# === LOAD & FILTER DATA =======================================
# ==============================================================
df = pd.read_csv(cohort_file)
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
if "training_level" in df.columns:
    df = df[df["training_level"] >= TRAINING_MIN]
if "session" in df.columns:
    df = df[df["session"] >= SESSION_MIN]


# ==============================================================
# === HELPER FUNCTIONS =========================================
# ==============================================================

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else x.std(ddof=1) / np.sqrt(len(x))


def prep_rt(df_in):
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
        .agg(mean=("mean_rt", "mean"), sem=("mean_rt", sem))
        .reset_index()
    )
    return per_subj, grouped


def prep_mt(df_in):
    per_subj = (
        df_in[df_in["success"] == 1]
        .groupby(["animal", "ABL", "ILD"])["timed_mt"]
        .agg(mean_mt="mean")
        .reset_index()
    )
    grouped = (
        per_subj.groupby(["ABL", "ILD"])
        .agg(mean=("mean_mt", "mean"), sem=("mean_mt", sem))
        .reset_index()
    )
    return per_subj, grouped


def prep_psy(df_in, do_individual_fits=False):
    all_pts = []
    per_subject_curves = {}
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
                    pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="my_psycho", n_trials=n_trials, show_plot=False
                    )
                    per_subject_curves[(subject, abl)] = dict(xx=xx, yy=yy)
                except Exception:
                    pass
    points = pd.DataFrame(all_pts)
    agg = (
        points.groupby(["ABL", "ILD"])
        .agg(mean=("PropLeft", "mean"), sem=("PropLeft", sem))
        .reset_index()
    )
    mean_fits = {}
    for abl in sorted(agg["ABL"].unique()):
        sub = agg[agg["ABL"] == abl]
        ILDs, y = sub["ILD"].values, sub["mean"].values
        n_trials = np.full_like(ILDs, 50)
        try:
            pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                ILDs, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
            mean_fits[abl] = dict(xx=xx, yy=yy)
        except Exception:
            mean_fits[abl] = None
    return points, agg, per_subject_curves, mean_fits



# ==============================================================
# === PREPARE VIEWS ============================================
# ==============================================================
prepared = {}
for view_name, make_view in views:
    df_v = make_view(df.copy())
    rt_per_subj, rt_group = prep_rt(df_v)
    mt_per_subj, mt_group = prep_mt(df_v)
    psy_points, psy_group, psy_indiv_curves, psy_mean_fits = prep_psy(
        df_v, do_individual_fits=(error_mode == "individuals")
    )
    prepared[view_name] = dict(
        rt_per_subj=rt_per_subj,
        rt_group=rt_group,
        mt_per_subj=mt_per_subj,
        mt_group=mt_group,
        psy_points=psy_points,
        psy_group=psy_group,
        psy_indiv_curves=psy_indiv_curves,
        psy_mean_fits=psy_mean_fits,
    )

abl_rows = sorted(set().union(*[set(p["rt_group"]["ABL"].unique()) for p in prepared.values()]))
view_colors = {
    name: preferred_view_colors.get(name, colors[i % len(colors)])
    for i, (name, _) in enumerate(views)
}


# ==============================================================
# === PLOTTING HELPERS =========================================
# ==============================================================
def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color="black")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE,
                   colors="black", width=1.5, length=6)
    for s in ["left", "right", "top", "bottom"]:
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_box_aspect(1)


def _plot_rt_row(ax, abl, tables, color, mode):
    rt_group = tables["rt_group"]
    rt_per_subj = tables["rt_per_subj"]
    if mode == "sem":
        sub = rt_group[rt_group["ABL"] == abl]
        if not sub.empty:
            x = DataHelpers.shift_ILD_for_ABL50(sub["ILD"])
            ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                        markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    else:
        for _, df_an in rt_per_subj[rt_per_subj["ABL"] == abl].groupby("animal"):
            ax.plot(DataHelpers.shift_ILD_for_ABL50(df_an["ILD"]), df_an["mean_rt"],
                    color=color, alpha=0.35, linewidth=1.5)
        sub = rt_group[rt_group["ABL"] == abl].sort_values("ILD")
        if not sub.empty:
            x = DataHelpers.shift_ILD_for_ABL50(sub["ILD"])
            ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                        markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
            ax.plot(x, sub["mean"], color=color, linewidth=2.5)


def _plot_mt_row(ax, abl, tables, color, mode):
    mt_group = tables["mt_group"]
    mt_per_subj = tables["mt_per_subj"]
    if mode == "sem":
        sub = mt_group[mt_group["ABL"] == abl]
        if not sub.empty:
            x = DataHelpers._ILD_for_ABL50(sub["ILD"])
            ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                        markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    else:
        for _, df_an in mt_per_subj[mt_per_subj["ABL"] == abl].groupby("animal"):
            ax.plot(DataHelpers.shift_ILD_for_ABL50(df_an["ILD"]), df_an["mean_mt"],
                    color=color, alpha=0.35, linewidth=1.5)
        sub = mt_group[mt_group["ABL"] == abl].sort_values("ILD")
        if not sub.empty:
            x = DataHelpers.shift_ILD_for_ABL50(sub["ILD"])
            ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o", color=color,
                        markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
            ax.plot(x, sub["mean"], color=color, linewidth=2.5)


def _plot_psy_row(ax, abl, tables, color, mode, skip_fit=None):
    psy_group = tables["psy_group"]
    psy_indiv = tables["psy_indiv_curves"]
    psy_mean = tables["psy_mean_fits"]

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(-19, 19)

    if mode == "sem":
        sub = psy_group[psy_group["ABL"] == abl]
        if not sub.empty:
            x_vals = DataHelpers.shift_ILD_for_ABL50(sub["ILD"])
            ax.errorbar(x_vals, sub["mean"], yerr=sub["sem"],
                        fmt="o", color=color, markersize=8.5,
                        linewidth=0, elinewidth=1.5, capsize=3)
        if not (skip_fit and (abl in skip_fit)):
            mean_fit = psy_mean.get(abl)
            if mean_fit:
                ax.plot(DataHelpers.shift_ILD_for_ABL50(mean_fit["xx"]), mean_fit["yy"],
                        color=color, linewidth=2)
    else:
        if not (skip_fit and (abl in skip_fit)):
            for (subject, abl_key), curve in psy_indiv.items():
                if abl_key != abl:
                    continue
                ax.plot(DataHelpers.shift_ILD_for_ABL50(curve["xx"]), curve["yy"],
                        color=color, alpha=0.3, linewidth=1)
        if not (skip_fit and (abl in skip_fit)):
            mean_fit = psy_mean.get(abl)
            if mean_fit:
                ax.plot(DataHelpers.shift_ILD_for_ABL50(mean_fit["xx"]), mean_fit["yy"],
                        color=color, linewidth=3, label="Avg sigmoid fit")
        sub = psy_group[psy_group["ABL"] == abl]
        if not sub.empty:
            x_vals = DataHelpers.shift_ILD_for_ABL50(sub["ILD"])
            ax.errorbar(x_vals, sub["mean"], yerr=sub["sem"],
                        fmt="o", color=color, markersize=8.5,
                        linewidth=0, elinewidth=1.5, capsize=3)


unique_abls = sorted(set().union(*[set(p["rt_group"]["ABL"].unique()) for p in prepared.values()]))
ABL_COLORS = {abl: f"C{i % 10}" for i, abl in enumerate(unique_abls)}


# ==============================================================
# === COMPUTE GROUP JNDs =======================================
# ==============================================================

# Collect JNDs from each subject in this view
all_jnds = []

# We'll use the same filtering logic as in prep_psy
for view_name, make_view in views:
    df_v = make_view(df.copy())

    for subject, df_subj in df_v.groupby("animal"):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="my_psycho")
        jnd_df = DataHelpers.compute_jnd_by_ABL(results, skip_ABL=50)
        if jnd_df.empty:
            continue
        jnd_df["subject"] = subject
        all_jnds.append(jnd_df)

# Combine all subjects’ JNDs
if len(all_jnds) > 0:
    all_jnds_df = pd.concat(all_jnds, ignore_index=True)

    # Compute mean ± SEM across subjects
    group_jnd = (
        all_jnds_df.groupby("ABL")["JND"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    print("\n[Group JND]")
    print(group_jnd)
else:
    print("[Group JND] No valid JNDs found!")
    group_jnd = pd.DataFrame(columns=["ABL", "mean", "sem"])



fig, axes = plt.subplots(len(views), 3, figsize=(22, 7 * len(views)), squeeze=False)

for r, (view_name, _) in enumerate(views):
    tables = prepared[view_name]
    abls = sorted(tables["rt_group"]["ABL"].unique())

    ax_rt, ax_mt, ax_psy = axes[r]
    for abl in abls:
        color = ABL_COLORS.get(abl, "gray")

        # --- RT ---
        _plot_rt_row(ax_rt, abl, tables, color, error_mode)


        # --- Overlay neurotypical (old) RT datapoints as squares ---
        # Draw the original (black) overlay behind everything
        DataHelpers.overlay_makefig1_rt(ax_rt, abl, makefig1_chrono, color="black", zorder=-1)

        # Re-extract the same old RT data to plot with custom color and marker
        try:
            out = DataHelpers.extract_rt_points(makefig1_chrono, abl)
            if out is not None:
                x_ref, y_ref, sem_ref = out
                x_ref = DataHelpers.shift_ILD_for_ABL50(x_ref)
                ax_rt.errorbar(
                    x_ref,
                    y_ref,
                    yerr=sem_ref,
                    fmt="s",                # square markers
                    color=color,            # same color as ABL curve
                    markersize=7,
                    linewidth=0,
                    elinewidth=1.5,
                    capsize=3,
                    alpha=1,
                    zorder=5,
                )
        except Exception as e:
            print(f"[warn] Could not overlay colored RT points for ABL {abl}: {e}")
        # --- MT ---
        _plot_mt_row(ax_mt, abl, tables, color, error_mode)

        # --- Psychometric ---
        _plot_psy_row(ax_psy, abl, tables, color, error_mode, skip_fit=SKIP_PSY_FITS)
        if abl !=50:
            DataHelpers.overlay_makefig1_psychometrics(ax_psy, makefig1_data, abl, color="black", show_individuals=False, use_abl_colors=False)



    # --- Axis styling ---
    _style_axes(ax_rt, f"{view_name} — RT", "ILD (dB)", "Mean RT (s)")
    _style_axes(ax_mt, f"{view_name} — MT", "ILD (dB)", "Mean MT (s)")
    _style_axes(ax_psy, f"{view_name} — Psychometric", "ILD (dB)", "P(Left)")

    # --- Tick relabeling for ±18 → ±50 ---
    for ax in [ax_rt, ax_mt, ax_psy]:
        xticks = sorted(set(ax.get_xticks()) | {-18, 18})
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [("-50" if x == -18 else "50" if x == 18 else str(int(x))) for x in xticks]
        )

# --- Global legend (colors per ABL) ---
handles, labels = [], []
for abl, color in ABL_COLORS.items():
    handles.append(plt.Line2D([], [], color=color, marker="o", linestyle="None"))
    labels.append(f"ABL {abl} dB")

fig.legend(
    handles, labels,
    loc="upper center",
    ncol=min(5, len(handles)),
    fontsize=LEGEND_FONTSIZE,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])

# --- Add JND inset ---

# Only include ABLs present in group_jnd
available_abls = sorted(group_jnd["ABL"].unique())

# Subset the color list to those ABLs (skip 50)
abl_color_map = ["C0", "C1", "C3"]

ax_inset = ax_psy.inset_axes([0.72, 0.15, 0.33, 0.33])  
# [x0, y0, width, height] in relative coordinates of the parent axis

for i, row in group_jnd.iterrows():
    abl = row["ABL"]
    color = abl_color_map[i % len(abl_color_map)]
    ax_inset.errorbar(
        abl, row["mean"],
        yerr=row["sem"],
        fmt="o",
        color=color,
        markersize=7,
        elinewidth=1.5,
        capsize=4,
        markeredgecolor="black",
        markeredgewidth=1,
    )

# Style the inset
_style_axes(ax_inset, title=None, xlabel="ABL", ylabel="JND (dB)")
ax_inset.set_xticks(sorted(group_jnd["ABL"].unique()))
ax_inset.tick_params(axis="both", labelsize=TICK_FONTSIZE - 6)
ax_inset.set_box_aspect(1)
ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)
ax_inset.grid(False)

ax_rt.set_xlim(0, 19)
ax_mt.set_xlim(-19, 19)
ax_psy.set_xlim(-19, 19)
plt.show()

#%%



# ==============================================================
# === NEW FIGURE: JND comparison (old vs new, per-animal) ======
# ==============================================================

# --- Load the original JND dataset from the old figure ---
old_jnd_path = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/jnd_analysis_data.pkl"
try:
    with open(old_jnd_path, "rb") as f:
        old_jnd_data = pickle.load(f)
except FileNotFoundError:
    print(f"Could not find {old_jnd_path}")
    old_jnd_data = None

if old_jnd_data is not None:
    old_ABLs = old_jnd_data["ABLS"]
    old_jnds = old_jnd_data["jnds"]
    old_animals = old_jnd_data["animals_with_mean"]

    # --- Define colors (ABL 60 uses C3) ---
    ABL_COLOR_MAP = {20: "C0", 40: "C1", 60: "C3"}

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # ==========================================================
    # OLD DATASET (open circles)
    # ==========================================================
    for abl in old_ABLs:
        color = ABL_COLOR_MAP.get(abl, "gray")
        if abl not in old_jnds:
            continue
        for animal in old_animals:
            if animal in old_jnds[abl]:
                ax.scatter(
                    abl - 0.5,  # slight left shift
                    old_jnds[abl][animal],
                    facecolors="none",
                    edgecolors=color,
                    s=60,
                    lw=1,
                    alpha=0.9,
                )

    # ==========================================================
    # NEW DATASET (closed circles, one per animal per ABL)
    # ==========================================================
    if len(all_jnds) > 0:
        for animal, df_an in all_jnds_df.groupby("subject"):
            for abl in sorted(df_an["ABL"].unique()):
                color = ABL_COLOR_MAP.get(abl, "gray")
                jnd_val = df_an.loc[df_an["ABL"] == abl, "JND"].values[0]
                ax.scatter(
                    abl + 0.5,  # slight right shift
                    jnd_val,
                    color=color,
                    s=55,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )
    """
    # ==========================================================
    # NEW GROUP MEANS ± SEM (optional overlay)
    # ==========================================================
    if not group_jnd.empty:
        for _, row in group_jnd.iterrows():
            color = ABL_COLOR_MAP.get(row["ABL"], "gray")
            ax.errorbar(
                row["ABL"] + 0.4, row["mean"],
                yerr=row["sem"],
                fmt="o", color=color,
                markersize=10, elinewidth=2,
                markeredgecolor="black", capsize=4,
                alpha=0.9, zorder=10,
            )
    """
    # ==========================================================
    # AXIS & STYLE
    # ==========================================================
    ax.set_xlabel("ABL (dB)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("JND (dB)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(sorted(ABL_COLOR_MAP.keys()))
    ax.set_xlim(15, 65)

    # Legend
    handles = [
        plt.Line2D([], [], marker="o", color="C0", linestyle="None", label="ABL 20"),
        plt.Line2D([], [], marker="o", color="C1", linestyle="None", label="ABL 40"),
        plt.Line2D([], [], marker="o", color="C3", linestyle="None", label="ABL 60"),
    ]
    #ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE - 2, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Skipping JND comparison — no old data found.")




#%%


"""
.##.....##.##.....##.##.......########.####....########...#######..##......##
.###...###.##.....##.##..........##.....##.....##.....##.##.....##.##..##..##
.####.####.##.....##.##..........##.....##.....##.....##.##.....##.##..##..##
.##.###.##.##.....##.##..........##.....##.....########..##.....##.##..##..##
.##.....##.##.....##.##..........##.....##.....##...##...##.....##.##..##..##
.##.....##.##.....##.##..........##.....##.....##....##..##.....##.##..##..##
.##.....##..#######..########....##....####....##.....##..#######...###..###.
"""

"""
GroupComparison (updated)
- Toggle between SEM error bars and per-subject overlays under the mean.
- Styling aligned with make_fig1.py.
- Conditional skipping of psychometric fit curves for selected ABLs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pickle
import Psychometric
import DataHelpers

# Old Data overlay

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

makefig1_data = load_makefig1_data("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_plot_data.pkl")
makefig1_chrono = load_makefig1_chrono("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_chrono_plot_data.pkl")


# --- Remap ABL 35 → 40 ---
if 35 in makefig1_data["ABLS"]:
    makefig1_data["ABLS"] = [40 if x == 35 else x for x in makefig1_data["ABLS"]]

    # Update all relevant dicts that use ABL as key
    for key in ["ilds_dict", "mean_sigmoid_dict", "mean_params_dict", "x_smooth_dict"]:
        if 35 in makefig1_data[key]:
            makefig1_data[key][40] = makefig1_data[key].pop(35)


if 35 in makefig1_chrono["plot_abls"]:
    makefig1_chrono["plot_abls"] = [40 if x == 35 else x for x in makefig1_chrono["plot_abls"]]

    # Update all relevant dicts that use ABL as key
    for key in ["ilds_dict", "mean_sigmoid_dict", "mean_params_dict", "x_smooth_dict"]:
        if 35 in makefig1_chrono[key]:
            makefig1_chrono[key][40] = makefig1_chrono[key].pop(35)

# =========================
# ====== CONFIG AREA ======
# =========================

# --- Paths / input file ---
# (leave as-is or override before running)
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2")
cohort_file = "merged_all_subjects.csv"

# --- Analysis choices ---
# error_mode: 'sem' or 'individuals'
#   'sem'         → colored mean ± SEM points
#   'individuals' → faint per-subject lines + thick black mean line/curve
error_mode = "individuals"             # <- change here: "sem" or "individuals" - of the current group

# ABL/level normalizations
MASK_59_TO_60 = True
MASK_25_TO_50_WHEN_TL16 = True
TRAINING_MIN = 16
SESSION_MIN  = 13

# Views
views = [
    ("Female wt", lambda d: DataHelpers.restrict_subjects(
        d, "sex_gen.csv", sex="female",  genotypes="wt",
        subject_col="animal", genotype_col="genotype", attach_meta=True)),


    ("Male wt",   lambda d: DataHelpers.restrict_subjects(
        d, "sex_gen.csv", sex="male", genotypes="wt",
        subject_col="animal", genotype_col="genotype", attach_meta=True)),
]

# --- Aesthetic choices copied from make_fig1.py ---
TITLE_FONTSIZE   = 24
LABEL_FONTSIZE   = 25
TICK_FONTSIZE    = 24
LEGEND_FONTSIZE  = 16
SUPTITLE_FONTSIZE = 24
TITLE_PAD        = 16  # extra space between title and axes

# Fonts & padding
mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'TeX Gyre Heros', 'Arial', 'sans-serif']

# Optional: print resolved font (helps debug environments)
try:
    font_path = fm.findfont(mpl.font_manager.FontProperties(family=mpl.rcParams['font.sans-serif']))
    print(f"[GroupComparison] Using font: {font_path}")
except Exception:
    pass

# Colors
colors = ["C0", "C1", "C2", "C3"]
FEMALE = "#e75480"   # pink
MALE   = "#1f77b4"   # blue

preferred_view_colors = {
    "Female wt": FEMALE,
    "Male wt":   MALE,
}

# Skip psychometric fit curves for specific ABLs (keep mean points ± SEM)
SKIP_PSY_FITS = {50}  # edit to set(), or e.g., {50, 60}

# =========================
# ====== LOAD / FILTER ====
# =========================
df = pd.read_csv(cohort_file)

if MASK_59_TO_60:
    mask = df["ABL"] == 59
    df.loc[mask, "ABL"] = 60

if MASK_25_TO_50_WHEN_TL16:
    mask = (df["training_level"] == 16) & (df["ABL"] == 25)
    df.loc[mask, "ABL"] = 50

if "training_level" in df.columns:
    df = df[df["training_level"] >= TRAINING_MIN]
if "session" in df.columns:
    df = df[df["session"] >= SESSION_MIN]

# =========================
# ===== PREPARATION =======
# =========================

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))

# 1) Reaction time tables
def prep_rt(df_in):
    # Keep only successful trials
    df_s = df_in[df_in["success"] == 1].copy()
    df_s["abs_ILD"] = df_s["ILD"].abs()

    # Mean within subject per (ABL, abs(ILD)), then aggregate across subjects
    per_subj = (
        df_s.groupby(["animal", "ABL", "abs_ILD"])["timed_rt"]
        .agg(mean_rt="mean")
        .reset_index()
        .rename(columns={"abs_ILD": "ILD"})  # <- critical line
    )

    # Group-level mean ± SEM
    grouped = (
        per_subj.groupby(["ABL", "ILD"]).agg(
            mean=("mean_rt", "mean"),
            sem=("mean_rt", sem),
            n=("mean_rt", "count")
        ).reset_index()
        .rename(columns={"abs_ILD": "ILD"})  # rename back for consistency
    )

    return per_subj, grouped


# 2) Movement time tables
def prep_mt(df_in):
    per_subj = (
        df_in[df_in["success"] == 1]
        .groupby(["animal","ABL","ILD"])["timed_mt"]
        .agg(mean_mt="mean")
        .reset_index()
    )
    grouped = (
        per_subj.groupby(["ABL","ILD"]).agg(
            mean=("mean_mt", "mean"),
            sem=("mean_mt", sem),
            n=("mean_mt", "count")
        ).reset_index()
    )
    return per_subj, grouped

# 3) Psychometric tables (+ optional per-subject fits for 'individuals' mode)
def prep_psy(df_in, do_individual_fits=False):
    # Per-subject points
    all_pts = []
    per_subject_curves = {}  # (animal, ABL) -> dict(xx, yy)

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
                    pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="my_psycho", n_trials=n_trials, show_plot=False
                    )
                    per_subject_curves[(subject, abl)] = dict(xx=xx, yy=yy)
                except Exception:
                    # Skip fit if anything goes off (robustness)
                    pass

    points = pd.DataFrame(all_pts)

    # Aggregate across subjects
    agg = (
        points.groupby(["ABL","ILD"]).agg(
            mean=("PropLeft", "mean"),
            sem=("PropLeft", sem),
            n=("PropLeft", "count")
        ).reset_index()
    )

    # Mean curve per ABL (for the thick mean, like make_fig1)
    mean_fits = {}
    for abl in sorted(agg["ABL"].unique()):
        sub = agg[agg["ABL"] == abl]
        ILDs = sub["ILD"].values
        y = sub["mean"].values
        n_trials = np.full_like(ILDs, 50)
        try:
            pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                ILDs, y, model="my_psycho", n_trials=n_trials, show_plot=False
            )
            mean_fits[abl] = dict(xx=xx, yy=yy)
        except Exception:
            mean_fits[abl] = None

    return points, agg, per_subject_curves, mean_fits

# =========================
# ======= PREP VIEWS ======
# =========================
prepared = {}  # view_name -> dict of tables
for view_name, make_view in views:
    df_v = make_view(df.copy())
    rt_per_subj, rt_group = prep_rt(df_v)
    mt_per_subj, mt_group = prep_mt(df_v)
    psy_points, psy_group, psy_indiv_curves, psy_mean_fits = prep_psy(
        df_v, do_individual_fits=(error_mode=="individuals")
    )
    prepared[view_name] = {
        "rt_per_subj": rt_per_subj, "rt_group": rt_group,
        "mt_per_subj": mt_per_subj, "mt_group": mt_group,
        "psy_points": psy_points, "psy_group": psy_group,
        "psy_indiv_curves": psy_indiv_curves, "psy_mean_fits": psy_mean_fits
    }

# Determine ABL rows present across views
abl_rows = sorted(set().union(*[set(p["rt_group"]["ABL"].unique()) for p in prepared.values()]))

# Assign colors to views (pink for female, blue for male)
view_colors = {}
for i, (name, _) in enumerate(views):
    view_colors[name] = preferred_view_colors.get(name, colors[i % len(colors)])

# =========================
# ======= PLOTTING ========
# =========================

def _style_axes(ax, title=None, xlabel=None, ylabel=None, square=True, hide_right_top=True):
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, color='black')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color='black')
    # Ticks: consistent size, color, and width
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, colors='black', width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=3)
    # Spines: all black and same width
    for side in ['left','right','top','bottom']:
        if side in ax.spines:
            ax.spines[side].set_color('black')
            ax.spines[side].set_linewidth(1.5)
    if hide_right_top:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    if square and hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1)

def _plot_rt_row(ax, abl, view_name, tables, color, mode):
    """Plot RT for one view in one ABL row onto ax."""
    rt_group = tables["rt_group"]
    rt_per_subj = tables["rt_per_subj"]

    if mode == "sem":
        sub = rt_group[rt_group["ABL"] == abl]
        if not sub.empty:
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt='o', color=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    else:  # individuals mode
        sub_ps = rt_per_subj[(rt_per_subj["ABL"] == abl)]
        for animal, df_an in sub_ps.groupby("animal"):
            df_an = df_an.sort_values("ILD")
            ax.plot(df_an["ILD"], df_an["mean_rt"], color=color, alpha=0.35, linewidth=1.5)
        sub = rt_group[rt_group["ABL"] == abl].sort_values("ILD")
        if not sub.empty:
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt='o', color=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3, zorder=3)
            ax.plot(sub["ILD"], sub["mean"], color=color, linewidth=2.5, zorder=2)

def _plot_mt_row(ax, abl, view_name, tables, color, mode):
    mt_group = tables["mt_group"]
    mt_per_subj = tables["mt_per_subj"]

    if mode == "sem":
        sub = mt_group[mt_group["ABL"] == abl]
        if not sub.empty:
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt='o', color=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)
    else:
        sub_ps = mt_per_subj[(mt_per_subj["ABL"] == abl)]
        for animal, df_an in sub_ps.groupby("animal"):
            df_an = df_an.sort_values("ILD")
            ax.plot(df_an["ILD"], df_an["mean_mt"], color=color, alpha=0.35, linewidth=1.5)
        sub = mt_group[mt_group["ABL"] == abl].sort_values("ILD")
        if not sub.empty:
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt='o', color=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3, zorder=3)
            ax.plot(sub["ILD"], sub["mean"], color=color, linewidth=2.5, zorder=2)

def _plot_psy_row(ax, abl, view_name, tables, color, mode, skip_fit=None):
    psy_group = tables["psy_group"]
    psy_indiv = tables["psy_indiv_curves"]
    psy_mean  = tables["psy_mean_fits"]

    # reference lines like make_fig1
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)

    if mode == "sem":
        sub = psy_group[psy_group["ABL"] == abl]
        if not sub.empty:
            x_vals = sub["ILD"].values.copy()
            if abl == 50:
                x_vals[:] = 18  # plot at 18 dB instead of real ILD
            ax.errorbar(x_vals, sub["mean"], yerr=sub["sem"],
                        fmt='o', color=color, markersize=8.5, linewidth=0,
                        elinewidth=1.5, capsize=3)
        # optional colored mean curve (keep group-colored to match the points)
        if not (skip_fit and (abl in skip_fit)):
            mean_fit = psy_mean.get(abl)
            if mean_fit:
                ax.plot(mean_fit["xx"], mean_fit["yy"], color=color, linewidth=2)
    else:
        # Individuals: faint fits per subject if present (unless skipped)
        if not (skip_fit and (abl in skip_fit)):
            for (subject, abl_key), curve in psy_indiv.items():
                if abl_key != abl: 
                    continue
                ax.plot(curve["xx"], curve["yy"], color=color, alpha=0.3, linewidth=1)
        # Group mean: thick black curve (unless skipped)
        if not (skip_fit and (abl in skip_fit)):
            mean_fit = psy_mean.get(abl)
            if mean_fit:
                ax.plot(mean_fit["xx"], mean_fit["yy"], color=color, linewidth=3, label='Avg sigmoid fit')
        # Also show mean points ± SEM in group color
        sub = psy_group[psy_group["ABL"] == abl]
        if not sub.empty:
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt='o', color=color, markersize=8.5, linewidth=0, elinewidth=1.5, capsize=3)


fig, axes = plt.subplots(len(abl_rows), 3, figsize=(18, 4.8*len(abl_rows)), squeeze=False)

for r, abl in enumerate(abl_rows):
    # RT
    ax_rt = axes[r, 0]
    for vi, (view_name, _) in enumerate(views):
        _plot_rt_row(ax_rt, abl, view_name, prepared[view_name], view_colors[view_name], error_mode)
    _style_axes(ax_rt, title=f"ABL {abl} — RT", xlabel="ILD (dB)", ylabel="Mean RT (s)")
    ax_rt.set_xscale('linear')
    # --- Overlay black RT curve from neurotypical animals ---
    DataHelpers.overlay_makefig1_rt(ax_rt, abl, makefig1_chrono)

    # MT
    ax_mt = axes[r, 1]
    for vi, (view_name, _) in enumerate(views):
        _plot_mt_row(ax_mt, abl, view_name, prepared[view_name], view_colors[view_name], error_mode)
    _style_axes(ax_mt, title=f"ABL {abl} — MT", xlabel="ILD (dB)", ylabel="Mean MT (s)")

    # Psychometric
    ax_psy = axes[r, 2]
    for vi, (view_name, _) in enumerate(views):
        _plot_psy_row(ax_psy, abl, view_name, prepared[view_name], view_colors[view_name], error_mode, skip_fit=SKIP_PSY_FITS)
    # --- Relabel x-axis so 18 appears as "50" (for shifted ABL 50 points) ---
    _style_axes(ax_psy, title=f"ABL {abl} — Psychometric", xlabel="ILD (dB)", ylabel="Proportion right")
    xticks = list(ax_psy.get_xticks())
    if 18 not in xticks:
        xticks.append(18)
    ax_psy.set_xticks(sorted(xticks))
    ax_psy.set_xticklabels([str(int(x)) if x != 18 else "50" for x in ax_psy.get_xticks()])
    
    if abl !=50:
        DataHelpers.overlay_makefig1_psychometrics(ax_psy, makefig1_data, abl, color="black", show_individuals=False, use_abl_colors=False)


# Global legend (top center)
handles = []
labels = []
for name in [v[0] for v in views]:
    h = plt.Line2D([], [], color=view_colors[name], marker='o', linestyle='None')
    handles.append(h)
    labels.append(name)
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), fontsize=LEGEND_FONTSIZE)

fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


 # %%
