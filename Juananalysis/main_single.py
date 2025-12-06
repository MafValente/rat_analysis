#%% SINGLE-SUBJECT PIPELINE — SETUP / COMPUTATION

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
sys.path.append("/Users/mafaldavalente/Documents/Mafalda_analysis/Juananalysis")
sys.path.append("/Users/mafaldavalente/Documents/Mafalda_analysis")

from load_data import load_behavior_csv
from plot_results import shaded_curve
from kernel_regression import (
    build_hierarchical_data_full,
    hierarchical_bootstrap_joint,
)
import DataHelpers

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
animal = "ASD0008"  # <- change animal here when needed

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
TRIALS_CSV = "merged_all_subjects.csv"
META_CSV   = "sex_gen.csv"

ABL_VALUES = [20, 40, 60]   # ABLs to plot for a single subject

COLORS_ABL = {
    20: "tab:blue",
    40: "tab:orange",
    60: "tab:green",
}

# X-axis limits for all plots
XMIN, XMAX = 0.0, 0.5   # <- change these if you like

# -------------------------------
# Load and preprocess
# -------------------------------
df = load_behavior_csv(TRIALS_CSV)
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"] == False].copy()

# Restrict to this subject
df = df[df["animal"] == animal].copy()

# Attach metadata (genotype, sex)
df = DataHelpers.restrict_subjects(
    df,
    meta_csv=META_CSV,
    sex=None,
    genotypes=None,
    attach_meta=True,
)

print(f"Analyzing subject: {animal}")
print("Genotype:", df["genotype"].unique())

# -------------------------------
# Kernel regression setup
# -------------------------------
xxi = np.linspace(0.0, 1.0, 200)
h   = 0.02
B   = 800  # 800 is usually enough for single animal

# -------------------------------
# STORAGE (same logic as before)
# -------------------------------
RTD   = {}
TCM   = {}
CDF   = {}
MTcur = {}

# -------------------------------
# Run bootstrap separately per ABL
# -------------------------------
for abl in ABL_VALUES:

    df_abl = df[df["ABL"] == abl].copy()

    if len(df_abl) == 0:
        RTD[abl]   = None
        TCM[abl]   = None
        CDF[abl]   = None
        MTcur[abl] = None
        continue

    data = build_hierarchical_data_full(
        df_abl,
        group_col="animal",
        group_value=animal,
        easy_value=None,
        abl_value=None,
    )

    RTD[abl], TCM[abl], CDF[abl], MTcur[abl] = hierarchical_bootstrap_joint(
        data, xxi, h, B
    )

print("✅ Finished kernel regression for single subject. You can now run the plotting cells.")
#%% PANEL 1 — Tachometric

fig, ax = plt.subplots(figsize=(6, 4))

for abl in ABL_VALUES:
    if TCM[abl] is None:
        continue
    med, up, dn = TCM[abl]
    shaded_curve(xxi, med, up, dn, color=COLORS_ABL[abl], label=f"ABL {abl}")

ax.set_xlabel("RT (s)")
ax.set_ylabel("P(correct)")
ax.set_ylim(0, 1)
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # vertical grid only
ax.set_title(f"Tachometric Curve — {animal}")
ax.legend()
plt.tight_layout()
plt.show()

#%% PANEL 2 — MT vs RT

fig, ax = plt.subplots(figsize=(6, 4))

for abl in ABL_VALUES:
    if MTcur[abl] is None:
        continue
    med, up, dn = MTcur[abl]
    shaded_curve(xxi, med, up, dn, color=COLORS_ABL[abl], label=f"ABL {abl}")

ax.set_xlabel("RT (s)")
ax.set_ylabel("MT (s)")
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # vertical grid only
ax.set_title(f"MT vs RT — {animal}")
ax.legend()
plt.tight_layout()
plt.show()

#%% PANEL 3 — RTD

fig, ax = plt.subplots(figsize=(6, 4))

for abl in ABL_VALUES:
    if RTD[abl] is None:
        continue
    med, up, dn = RTD[abl]
    shaded_curve(xxi, med, up, dn, color=COLORS_ABL[abl], label=f"ABL {abl}")

ax.set_xlabel("RT (s)")
ax.set_ylabel("RT density")
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # vertical grid only
ax.set_title(f"RTD — {animal}")
ax.legend()
plt.tight_layout()
plt.show()

#%% PANEL 4 — CDF

fig, ax = plt.subplots(figsize=(6, 4))

for abl in ABL_VALUES:
    if CDF[abl] is None:
        continue
    med, up, dn = CDF[abl]
    shaded_curve(xxi, med, up, dn, color=COLORS_ABL[abl], label=f"ABL {abl}")

ax.set_xlabel("RT (s)")
ax.set_ylabel("Cumulative probability")
ax.set_ylim(0, 1)
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # vertical grid only
ax.set_title(f"CDF — {animal}")
ax.legend()
plt.tight_layout()
plt.show()
