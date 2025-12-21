
#%%
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
import Helpers.DataHelpers as DataHelpers

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

makefig1_data = load_makefig1_data("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ILD_task/fig1_plot_data.pkl")
makefig1_chrono = load_makefig1_chrono("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ILD_task/fig1_chrono_plot_data.pkl")
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
error_mode = "individuals"             # <- change here: "sem" or "individuals"

# ABL/level normalizations
MASK_59_TO_60 = True
MASK_25_TO_50_WHEN_TL16 = True
TRAINING_MIN = 16
SESSION_MIN  = 13

# Views
views = [
    ("wt", lambda d: DataHelpers.restrict_subjects(
        d, "sex_gen.csv",  genotypes="wt",
        subject_col="animal", genotype_col="genotype", attach_meta=True)),
]
"""
    ("Male wt",   lambda d: DataHelpers.restrict_subjects(
        d, "sex_gen.csv", sex="male", genotypes="wt",
        subject_col="animal", genotype_col="genotype", attach_meta=True)),
"""

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
    "Females": FEMALE,
    "Males":   MALE,
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
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="erf_4par")
        for abl, res in results.items():
            ILDs = np.asarray(res["ILDs"])
            pleft = np.asarray(res["PropLeft"], dtype=float)
            for ild, val in zip(ILDs, pleft):
                all_pts.append({"subject": subject, "ABL": abl, "ILD": ild, "PropLeft": val})

            if do_individual_fits and len(ILDs) >= 4 and np.isfinite(pleft).all():
                n_trials = np.full_like(ILDs, 50)
                try:
                    pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
                        ILDs, pleft, model="erf_4par", n_trials=n_trials, show_plot=False
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
                ILDs, y, model="erf_4par", n_trials=n_trials, show_plot=False
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

    DataHelpers.overlay_makefig1_psychometrics(ax_psy, abl, makefig1_data)
    for vi, (view_name, _) in enumerate(views):
        _plot_psy_row(ax_psy, abl, view_name, prepared[view_name], view_colors[view_name], error_mode, skip_fit=SKIP_PSY_FITS)
    # --- Relabel x-axis so 18 appears as "50" (for shifted ABL 50 points) ---
    xticks = list(ax_psy.get_xticks())
    if 18 not in xticks:
        xticks.append(18)
    ax_psy.set_xticks(sorted(xticks))
    ax_psy.set_xticklabels([str(int(x)) if x != 18 else "50" for x in ax_psy.get_xticks()])




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
# Assign fixed colors per ABL (consistent across views)
unique_abls = sorted(set().union(*[set(p["rt_group"]["ABL"].unique()) for p in prepared.values()]))
ABL_COLORS = {abl: f"C{i % 10}" for i, abl in enumerate(unique_abls)}

fig, axes = plt.subplots(len(views), 3, figsize=(22, 7 * len(views)), squeeze=False)


for r, (view_name, _) in enumerate(views):
    tables = prepared[view_name]
    abls = sorted(tables["rt_group"]["ABL"].unique())

    # --- RT ---
    ax_rt = axes[r, 0]
    for abl in abls:
        color = ABL_COLORS.get(abl, "gray")
        _plot_rt_row(ax_rt, abl, view_name, tables, color=color, mode=error_mode)
    _style_axes(ax_rt, title=f"{view_name} — RT", xlabel="ILD (dB)", ylabel="Mean RT (s)")
    ax_rt.set_xscale("linear")
    DataHelpers.overlay_makefig1_rt(ax_rt, "all", makefig1_chrono)

    # --- MT ---
    ax_mt = axes[r, 1]
    for abl in abls:
        color = ABL_COLORS.get(abl, "gray")
        _plot_mt_row(ax_mt, abl, view_name, tables, color=color, mode=error_mode)
    _style_axes(ax_mt, title=f"{view_name} — MT", xlabel="ILD (dB)", ylabel="Mean MT (s)")

    # --- Psychometric ---
    ax_psy = axes[r, 2]
    for abl in abls:
        color = ABL_COLORS.get(abl, "gray")
        _plot_psy_row(ax_psy, abl, view_name, tables, color=color, mode=error_mode, skip_fit=SKIP_PSY_FITS)
        DataHelpers.overlay_makefig1_psychometrics(ax_psy, abl, makefig1_data)
    # --- Relabel x-axis so 18 appears as "50" (for shifted ABL 50 points) ---
    xticks = list(ax_psy.get_xticks())
    if 18 not in xticks:
        xticks.append(18)
    ax_psy.set_xticks(sorted(xticks))
    ax_psy.set_xticklabels([str(int(x)) if x != 18 else "50" for x in ax_psy.get_xticks()])


# Global legend (colors = ABLs)
handles, labels = [], []
for abl, color in ABL_COLORS.items():
    h = plt.Line2D([], [], color=color, marker='o', linestyle='None')
    handles.append(h)
    labels.append(f"ABL {abl} dB")

fig.legend(handles, labels, loc="upper center", ncol=min(5, len(handles)), fontsize=LEGEND_FONTSIZE)
fig.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# %%
