#%% ==========================================================
# Compare 4-parameter psychometric fits between merged_all_subjects
# and merged_valid (from fig1_plot_data.pkl)
#==============================================================

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Psychometric import compute_psychometrics_by_ABL  
from scipy.optimize import curve_fit
import Helpers.DataHelpers as DataHelpers

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


## ================================================================
# --- Load and restrict data ---
# ================================================================

meta = DataHelpers._load_subject_metadata("sex_gen.csv")  # path to your metadata file
meta = meta.rename(columns={"subject": "animal"})   # ensure merge key matches

cohort_file = "merged_all_subjects.csv"
df_ASD2 = pd.read_csv(cohort_file)

# Load merged_valid from pickle
with open("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_plot_data.pkl", "rb") as f:
    data = pickle.load(f)
df_NT = data["merged_valid"]



# ==============================================================
# === FILTER DATA ==============================================
# ==============================================================
# Keep only ABL 20, 40, 60
ABLs_to_use = [20, 40, 60]

df_NT = df_NT[df_NT["ABL"].isin(ABLs_to_use)].copy()

MASK_59_TO_60 = True
MASK_25_TO_50_WHEN_TL16 = True
TRAINING_MIN = 16
SESSION_MIN = 13

df_ASD2 = DataHelpers.prepare_data(df_ASD2, session_col="session", trial_col="trial")
if "training_level" in df_ASD2.columns:
    df_ASD2 = df_ASD2[df_ASD2["training_level"] >= TRAINING_MIN]
if "session" in df_ASD2.columns:
    df_ASD2 = df_ASD2[df_ASD2["session"] >= SESSION_MIN]

df_ASD2 = df_ASD2[df_ASD2["ABL"].isin(ABLs_to_use)].copy()

df_ASD2 = df_ASD2[df_ASD2["ABL"].isin(ABLs_to_use)].copy()

print(f"Dataset filtered to ABLs {ABLs_to_use}")
print(f"  merged_all_subjects: {df_ASD2.shape[0]} trials")
print(f"  merged_valid: {df_NT.shape[0]} trials")

# ================================================================
# --- Robust fitting across animals with error handling ---
# ================================================================
def compute_all_animals(df, label):
    """Run psychometric fits for each animal × ABL using my_psycho model."""
    all_results = []
    for animal in sorted(df["animal"].unique()):
        df_animal = df[df["animal"] == animal]
        try:
            results = compute_psychometrics_by_ABL(df_animal, model="my_psycho")
        except Exception as e:
            print(f"⚠️  Skipping {animal}: failed ({e})")
            continue

        for abl, res in results.items():
            pars = res.get("pars", [np.nan]*4)
            if any(np.isnan(pars)):
                print(f"   ⚠️  {animal}, ABL={abl}: fit failed, keeping NaN")


            all_results.append({
                "animal": animal,
                "ABL": abl,
                "source": label,
                "slope_a": pars[0],
                "bias_b": pars[1],
                "lower_c": pars[2],
                "upper_d": pars[3],
            })
    return pd.DataFrame(all_results)


print("\n--- Fitting merged_all_subjects ---")
df_ASD2_params = compute_all_animals(df_ASD2, "merged_all_subjects")

print("\n--- Fitting merged_NT ---")
df_NT_params = compute_all_animals(df_NT, "merged_NT")

# Combine
df_params = pd.concat([df_ASD2_params, df_NT_params], ignore_index=True)
df_params["source"] = df_params["source"].replace("merged_NT", "merged_valid")

df_params["animal"] = df_params["animal"].astype(str)
# --- Merge genotype info into df_params before using it ---
df_params = df_params.merge(meta.rename(columns={"subject": "animal"}), on="animal", how="left")

df_params["genotype"] = df_params["genotype"].fillna("unknown")

print("\nExtracted parameters:")
print(df_params.head())

animals_sorted = sorted(df_params["animal"].unique())


# ================================================================
# --- Plotting 4 panels (one per parameter) ---
# ================================================================

# --- Ensure animals are strings for plotting ---
df_params["animal"] = df_params["animal"].astype(str)

# --- Define colors per ABL (match your figure convention) ---
ABL_colors = {20: "C0", 40: "C1", 60: "C3"}

params = ["slope_a", "bias_b", "lower_c", "upper_d"]
titles = ["Slope (a)", "Bias (b)", "Lower (c)", "Upper (d)"]
# Define markers per genotype
marker_map = {"wt": "o", "het": "s", "hom": "^", "unknown": "o"}

fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True)
axes = axes.ravel()   # flatten 2×2 array to 1-D list for easy looping

animals_sorted = sorted(df_params["animal"].unique())

for i, (param, title) in enumerate(zip(params, titles)):
    ax = axes[i]
    """
    # --- FILLED markers: ASD cohort (merged_all_subjects) numerically sorted ---
    for abl, color in ABL_colors.items():
        data_all = df_params[
            (df_params["source"] == "merged_all_subjects") &
            (df_params["ABL"] == abl)
        ]
        # Plot each point individually so order is preserved
        for _, row in data_all.iterrows():
            marker = marker_map.get(row["genotype"], "o")
            ax.scatter(
                row["animal"], row[param],
                marker=marker,
                s=80,
                facecolor=color, edgecolor="black",
                linewidth=0.6, zorder=3,
                label=f"{row['genotype'].upper()} – ABL {abl} (ASD)"
            )
    """

        # --- FILLED markers: ASD cohort (merged_all_subjects) ---
    for abl, color in ABL_colors.items():
        for genotype, marker in marker_map.items():
            data_all = df_params[
                (df_params["source"] == "merged_all_subjects")
                & (df_params["ABL"] == abl)
                & (df_params["genotype"] == genotype)
            ]
            if data_all.empty:
                continue
            ax.scatter(
                data_all["animal"], data_all[param],
                marker=marker,
                s=80,
                facecolor=color, edgecolor="black",
                linewidth=0.6, zorder=3,
                label=f"{genotype.upper()} – ABL {abl} (ASD)"
            )


   # --- OPEN markers: NT cohort (merged_valid) ---
    for abl, color in ABL_colors.items():
        data_valid = df_params[
            (df_params["source"] == "merged_valid")
            & (df_params["ABL"] == abl)
        ]
        if data_valid.empty:
            continue
        ax.scatter(
            data_valid["animal"], data_valid[param],
            marker="o",
            s=100,
            facecolors="none", edgecolors=color,
            linewidth=1.8, zorder=4,
            label=f"ABL {abl} (NT)"
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Animal")
    if i == 0:
        ax.set_ylabel("Parameter value")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    if title == "Bias (b)":
        ax.set_ylim(-3, 3)


# ================================================================
# --- Compact legend ---
# ================================================================
from matplotlib.lines import Line2D

# ABL colour symbols (filled)
abl_handles = [
    Line2D([], [], marker="o", color="black", markerfacecolor=c, markersize=8, linestyle="None")
    for c in ABL_colors.values()
]
abl_labels = [f"ABL {a}" for a in ABL_colors.keys()]

# Genotype symbols (black only)
geno_handles = [
    Line2D([], [], marker="o", color="black", markersize=8, linestyle="None", markerfacecolor="black"),
    Line2D([], [], marker="s", color="black", markersize=8, linestyle="None", markerfacecolor="black"),
    Line2D([], [], marker="^", color="black", markersize=8, linestyle="None", markerfacecolor="black"),
]
geno_labels = ["wt","het", "hom"]

# Dataset symbol (open circle)
open_handle = Line2D([], [], marker="o", color="black", markerfacecolor="none",
                     markersize=8, linestyle="None", markeredgewidth=1.2)
open_label = ["NT cohort"]

# Combine
handles = abl_handles + geno_handles + [open_handle]
labels = abl_labels + geno_labels + open_label

axes[0].legend(handles, labels, frameon=False, fontsize=9, loc="best", ncol=1)

plt.tight_layout()
plt.show()

# ================================================================
# --- Save results ---
# ================================================================
#df_params.to_csv("psychometric_parameters_by_animal.csv", index=False)
#print("\nSaved results to psychometric_parameters_by_animal.csv")

#%% ================================================================
