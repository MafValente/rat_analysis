#%%

"""
..######..####.##....##..######...##.......########...
.##....##..##..###...##.##....##..##.......##.........
.##........##..####..##.##........##.......##.........
..######...##..##.##.##.##...####.##.......######.....
.......##..##..##..####.##....##..##.......##.........
.##....##..##..##...###.##....##..##.......##.........
..######..####.##....##..######...########.########...
"""

# ===============================================================
#   IMPORTS
# ===============================================================
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import DataHelpers
import numpy as np

# ===============================================================
#   Setup & data loading
# ===============================================================

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")

subject_file = "merged_ASD0021.csv"
subject_id = subject_file.removeprefix("merged_").removesuffix(".csv")

df = pd.read_csv(subject_file)
df = df[df["training_level"] ==16]
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")

df_valid = df[df["abort_type"]!="CNP"].copy()

mask_aborts = (
    (df["abort_type"] == "Fixation")
)
df_aborts = df[mask_aborts].copy()

# ===============================================================
#   Global style settings
# ===============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "axes.linewidth": 1.5,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ===============================================================
#   Bin Widths
# ===============================================================
bin_width_ft = 20   # <-- this is in ms
bin_width_cnp = .5   # <-- CHANGE THIS to whatever bin size you need
bin_width = 0.01   # <-- CHANGE THIS to whatever bin size you need
bin_width_mt = 0.01   # <-- CHANGE THIS to whatever bin size you need
bin_width_lnp = 0.001   # <-- CHANGE THIS to whatever bin size you need

# ===============================================================
#   Histogram
# ===============================================================

cols = ["fix_time", "intended_fix_time", "cnp_time"]
cols2 = ["timed_mt", "timed_lnp"]

fig, axes = plt.subplots(2, len(cols), figsize=(12, 8))  # 2 rows × 3 columns

# --- Row 1: all from df_valid ---

ft_min = df_valid["fix_time"].min()
ft_max = df_valid["fix_time"].max()

cnp_min = df_valid["cnp_time"].min()
cnp_max = df_valid["cnp_time"].max()

# Create fixed bin edges
bins_ft = np.arange(0, ft_max + bin_width_ft, bin_width_ft)
bins_cnp = np.arange(cnp_min, cnp_max + bin_width_cnp, bin_width_cnp)
binscols = [bins_ft, bins_ft, bins_cnp]

for ax, col, bins in zip(axes[0, :], cols, binscols):
    ax.hist(
        df_valid[col].dropna(),
        bins=bins,
        histtype="step",
        color="black",
        linewidth=1,
        label="All trials",
    )

     # === Add second histogram only for the FIRST PANEL ===
    if col == "fix_time":
        ax.hist(
            df_aborts["fix_time"].dropna(),
            bins=bins_ft,            
            histtype="step",
            color="red",
            linewidth=1,
            label="Fixation aborts",
        )
        ax.legend(fontsize=10)

    # choose correct bin to print width based on column
    if col in ["fix_time", "intended_fix_time"]:
        bw = bin_width_ft
    elif col == "cnp_time":
        bw = bin_width_cnp

    # annotate in top-left of panel
    ax.text(
        0.75, 0.95,
        f"bin = {bw}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10
    )

    # your aesthetic
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color("black")
    ax.tick_params(axis="both", labelsize=11, width=1.3)

# --- Row 2, col 0: from df ---

# --- Set fixed bin width ---

rt_min = df["timed_rt"].min()
rt_max = df["timed_rt"].max()

# Create fixed bin edges
bin_rt = np.arange(rt_min, rt_max + bin_width, bin_width)

df["timed_rt"].hist(
    bins=bin_rt,
    color="lightblue",
    histtype="step", 
    edgecolor="black",
    ax=axes[1, 0],
)

# --- Row 2, col 1–2: from df_valid ---

mt_min = df_valid["timed_mt"].min()
mt_max = df_valid["timed_mt"].max()

lnp_min = df_valid["timed_lnp"].min()
lnp_max = df_valid["timed_lnp"].max()

# Create fixed bin edges
bins_mt = np.arange(mt_min, mt_max + bin_width_mt, bin_width_mt)
bins_lnp = np.arange(lnp_min, lnp_max + bin_width_lnp, bin_width_lnp)
binscols2 = [bins_mt, bins_lnp]

for ax, col, bins in zip(axes[1, 1:3], cols2, binscols2):

    data_source = df_valid
    data = data_source[col].dropna()

    ax.hist(
        data,
        bins=bins,
        histtype="step",
        linewidth=1,
        color="black"
    )
    # choose correct bin width
    if col == "timed_mt":
        bw = bin_width_mt
    elif col == "timed_lnp":
        bw = bin_width_lnp

    ax.text(
        0.75, 0.95,
        f"bin = {bw}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10
    )

# --- Shared styling ---
row_cols = [cols, ["timed_rt", "timed_mt", "timed_lnp"]]
for ax_row, colset in zip(axes, row_cols):
    for col, ax in zip(colset, ax_row):
        xlabel_map = {
            "fix_time": "Fixation Time (ms)",
            "intended_fix_time": "Intended Fixation Time (ms)",
            "cnp_time": "CNP Time (s)",
            "timed_rt": "Timed RT (s)",
            "timed_mt": "Timed MT (s)",
            "timed_lnp": "Timed LNP (s)",
        }
        ax.set_xlabel(xlabel_map.get(col, col), fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color("black")
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.3, color="black")
        ax.grid(False)

# Example of per-plot axis limits (optional)
axes[1, 0].set_xlim(-1, 1.5)
axes[1, 1].set_xlim(0, 1)
axes[0, 2].set_xlim(-1, 10)
axes[0, 0].set_xlim(-2, 2000)
axes[1, 0].set_xlim(-.5, .5)
axes[1, 2].set_xlim(0, .02)

ax = axes[1,0]
ax.text(
    0.75, 0.95,
    f"bin = {bin_width}",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=10
)

plt.suptitle(f"{subject_id}: Timing Distributions", fontsize=15, y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# --- Prepare dataframe ---
df_rt = df_valid.dropna(subset=["timed_rt", "ABL"]).copy()
df_rt = df_rt[df_rt["ABL"] != 50]   # exclude ABL = 50

# --- Set fixed bin width ---
bin_width = 0.01   # <-- CHANGE THIS to whatever bin size you need
rt_min = df_rt["timed_rt"].min()
rt_max = df_rt["timed_rt"].max()

# Create fixed bin edges
bins = np.arange(rt_min, rt_max + bin_width, bin_width)

# --- Plot ---
plt.figure(figsize=(4, 2.5))

for abl in sorted(df_rt["ABL"].unique()):
    subset = df_rt[df_rt["ABL"] == abl]["timed_rt"]

    plt.hist(
        subset,
        bins=bins,
        histtype="step",     # stairs outline
        linewidth=1.5,
        label=f"ABL {abl}"
    )

plt.xlabel("Reaction Time (s)", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.title(f"{subject_id}: Reaction Times by ABL", fontsize=15, fontweight="bold")

# Style (matching your global settings)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color("black")
ax.tick_params(axis="both", labelsize=11, width=1.3)

plt.xlim(0, .75)      # set x-axis limits (example)


plt.legend(title="ABL", fontsize=11)
plt.tight_layout()
plt.show()








# %%

"""
..######...########...#######..##.....##.########.
.##....##..##.....##.##.....##.##.....##.##.....##
.##........##.....##.##.....##.##.....##.##.....##
.##...####.########..##.....##.##.....##.########.
.##....##..##...##...##.....##.##.....##.##.......
.##....##..##....##..##.....##.##.....##.##.......
..######...##.....##..#######...#######..##.......
"""
# ===============================================================
#   IMPORTS
# ===============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataHelpers   # your module
import os
import seaborn as sns

# ===============================================================
#   GLOBAL BIN WIDTHS FOR ALL COLUMNS
# ===============================================================
BIN_WIDTHS = {
    "fix_time": 20,              # ms
    "intended_fix_time": 20,     # ms
    "cnp_time": 0.1,             # s
    "timed_rt": 0.01,            # s
    "timed_mt": 0.01,            # s
    "timed_lnp": 0.0001,          # s
}


# ===============================================================
#   GENOTYPE VIEWS (YOUR DEFINITION)
# ===============================================================
def get_views(meta_csv):
    return [
        ("wt", lambda d: DataHelpers.restrict_subjects(
            d, meta_csv, genotypes="wt", subject_col="animal",
            genotype_col="genotype", attach_meta=True)),

        ("het", lambda d: DataHelpers.restrict_subjects(
            d, meta_csv, genotypes="het", subject_col="animal",
            genotype_col="genotype", attach_meta=True)),

        ("hom", lambda d: DataHelpers.restrict_subjects(
        d, meta_csv, genotypes="hom", subject_col="animal",
        genotype_col="genotype", attach_meta=True)),
    ]


# ===============================================================
#   (1) COMPUTE PER-ANIMAL HISTOGRAM
# ===============================================================
def compute_hist_for_animal(df, column, bins, animal_col):
    data = df[column].dropna().values
    counts, _ = np.histogram(data, bins=bins)

    # Normalize so each animal contributes equally (PDF)
    total = counts.sum()
    if total > 0:
        counts = counts / total

    return counts


# ===============================================================
#   (2) BUILD GROUP MATRIX OF HISTOGRAMS
# ===============================================================
def build_group_matrix(df, column, bins, animal_col):
    animals = df[animal_col].unique()
    mat = []

    for animal in animals:
        dfa = df[df[animal_col] == animal]
        counts = compute_hist_for_animal(dfa, column, bins, animal_col)
        mat.append(counts)

    return animals, np.array(mat)   # shape = (n_animals, n_bins-1)


# ===============================================================
#   (3) MEAN + SEM ACROSS ANIMALS
# ===============================================================
def mean_sem(mat):
    mean = mat.mean(axis=0)
    sem  = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    return mean, sem


# ===============================================================
#   (4) MULTI-PANEL PLOT OF GROUP CURVES
# ===============================================================
def plot_group_panels(group_results, bins_dict, subject_id, colors):
    cols_row1 = ["fix_time", "intended_fix_time", "cnp_time"]
    cols_row2 = ["timed_rt", "timed_mt", "timed_lnp"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # ----- ROW 1 -----
    for ax, col in zip(axes[0, :], cols_row1):
        for group_name, results in group_results.items():
            mids, mean, sem = results[col]
            ax.step(mids, mean, color=colors[group_name], where='mid',
                    linewidth=1.8, label=group_name)
            ax.fill_between(mids, mean-sem, mean+sem, step='mid',
                            alpha=0.25, color=colors[group_name])

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(False)

    # ----- ROW 2 (timed_rt gets column 0) -----
    col_rt = cols_row2[0]
    ax = axes[1, 0]
    for group_name, results in group_results.items():
        mids, mean, sem = results[col_rt]
        ax.step(mids, mean, color=colors[group_name], where='mid',
                linewidth=1.8, label=group_name)
        ax.fill_between(mids, mean-sem, mean+sem, step='mid',
                        alpha=0.25, color=colors[group_name])
    ax.set_title(col_rt)
    ax.set_xlabel(col_rt)
    ax.set_ylabel("Density")
    ax.grid(False)

    # ----- ROW 2 (timed_mt & timed_lnp) -----
    for ax, col in zip(axes[1, 1:], cols_row2[1:]):
        for group_name, results in group_results.items():
            mids, mean, sem = results[col]
            ax.step(mids, mean, color=colors[group_name], where='mid',
                    linewidth=1.8, label=group_name)
            ax.fill_between(mids, mean-sem, mean+sem, step='mid',
                            alpha=0.25, color=colors[group_name])

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(False)

    axes[0, 0].legend()

    axes[1, 0].set_xlim(-1, 1.5)
    axes[1, 1].set_xlim(0, 1)
    axes[0, 2].set_xlim(-1, 10)
    axes[0, 0].set_xlim(-2, 2000)
    axes[1, 0].set_xlim(-.5, .5)
    axes[1, 2].set_xlim(0.01, .014)
    plt.suptitle(f"{subject_id}: Group Mean Timing Histograms", fontsize=16)
    plt.tight_layout()
    plt.show()


# ===============================================================
#   (5) TOP-LEVEL ANALYSIS FUNCTION
# ===============================================================
def analyze_groups(
    merged_all,
    views,
    timing_columns,
    meta_csv,
    animal_col="animal",
    subject_id="Group Summary"
):

    # ---- build bins ----
    bins_dict = {}
    for col in timing_columns:
        bw = BIN_WIDTHS[col]
        col_min = merged_all[col].min()
        col_max = merged_all[col].max()
        bins_dict[col] = np.arange(col_min, col_max + bw, bw)

    # ---- compute histograms for each group ----
    group_results = {}


    for group_name, get_group in views:
        df_group = get_group(merged_all)   # restrict subjects

        print(f"\n=== {group_name.upper()} ===")
        print("Animals in this group:")
        print(df_group["animal"].unique())
        print(f"Total animals: {df_group['animal'].nunique()}")


        results_for_group = {}
        for col in timing_columns:
            bins = bins_dict[col]
            animals, mat = build_group_matrix(df_group, col, bins, animal_col)
            mean, sem = mean_sem(mat)
            mids = 0.5*(bins[:-1] + bins[1:])
            results_for_group[col] = (mids, mean, sem)

        group_results[group_name] = results_for_group

        print("DEBUG groups:", list(group_results.keys()))

    group_names = list(group_results.keys())
    palette = sns.color_palette("Set1", len(group_names))
    colors = {g: palette[i] for i,g in enumerate(group_names)}


    # ---- plot all panels ----
    plot_group_panels(group_results, bins_dict, subject_id, colors)

#%%
# ===============================================================
#   MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":


    # Load merged_all
    os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
    df = pd.read_csv("merged_all_subjects.csv")
    # path to your meta file
    META_CSV = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2/sex_gen.csv"


    # preprocessing
    df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
    df = df[df["training_level"] == 16].copy()
    df = df[df["abort_type"]!="CNP"].copy()

    # which timing columns to analyze
    timing_columns = [
        "fix_time",
        "intended_fix_time",
        "cnp_time",
        "timed_rt",
        "timed_mt",
        "timed_lnp",
    ]

    views = get_views(META_CSV)

    analyze_groups(
        merged_all=df,
        views=views,
        timing_columns=timing_columns,
        meta_csv=META_CSV,
        animal_col="animal",
        subject_id="All Animals",
    )

#%%

def compute_group_abl_rt_views(merged_all, views, bins, animal_col="animal"):
    """
    Computes RT distributions per genotype (from views)
    and per ABL = 20, 40, 60.
    """

    ABL_values = [20, 40, 60]
    results = {}

    for group_name, get_group in views:

        df_group = get_group(merged_all)    # same logic as first figure
        results[group_name] = {}

        for abl in ABL_values:
            df_ga = df_group[df_group["ABL"] == abl]

            animals = df_ga[animal_col].unique()
            mats = []

            for animal in animals:
                dfa = df_ga[df_ga[animal_col] == animal]
                data = dfa["timed_rt"].dropna().values
                counts, _ = np.histogram(data, bins=bins)

                # normalization per-animal
                s = counts.sum()
                if s > 0:
                    counts = counts / s

                mats.append(counts)

            if len(mats) == 0:
                results[group_name][abl] = None
                continue

            mats = np.array(mats)
            mean = mats.mean(axis=0)
            sem = mats.std(axis=0, ddof=1) / np.sqrt(mats.shape[0])
            mids = 0.5*(bins[:-1] + bins[1:])

            results[group_name][abl] = (mids, mean, sem)

    return results

def plot_rt_by_abl(results, bins, subject_id):
    """
    One panel per genotype.
    Inside each panel: RT for ABL 20, 40, 60.
    """

    genos = list(results.keys())
    fig, axes = plt.subplots(1, len(genos), figsize=(5*len(genos), 4), sharey=True)

    if len(genos) == 1:
        axes = [axes]

    colors = {20: "red", 40: "blue", 60: "green"}

    for ax, geno in zip(axes, genos):
        for abl in [20, 40, 60]:

            item = results[geno].get(abl)
            if item is None:
                continue

            mids, mean, sem = item

            ax.step(mids, mean, where="mid", color=colors[abl],
                    label=f"ABL {abl}", linewidth=1.8)
            ax.fill_between(
                mids,
                mean - sem,
                mean + sem,
                step="mid",
                alpha=0.25,
                color=colors[abl]
            )

        ax.set_title(geno.upper())
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
        ax.legend()
        axes[0].set_xlim(0, .7)
        axes[1].set_xlim(0, .7)
        axes[2].set_xlim(0, .7)


    plt.suptitle(f"RT Distributions by ABL (20, 40, 60)\n{subject_id}", fontsize=15)
    plt.tight_layout()
    plt.show()

# ----- build bins for RT -----
bw = BIN_WIDTHS["timed_rt"]
rt_min = df["timed_rt"].min()
rt_max = df["timed_rt"].max()
bins_rt = np.arange(rt_min, rt_max + bw, bw)

# ----- compute per-genotype per-ABL RT histograms -----
results_rt = compute_group_abl_rt_views(df, views, bins_rt)


# ----- plot -----
plot_rt_by_abl(results_rt, bins_rt, subject_id="All Animals")

# %%
