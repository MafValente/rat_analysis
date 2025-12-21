#%%
# GROUP-LEVEL ANALYSIS PIPELINE — SETUP / COMPUTATION

import sys
sys.path.append("/Users/mafaldavalente/Documents/Mafalda_analysis/Juananalysis")
sys.path.append("/Users/mafaldavalente/Documents/Mafalda_analysis")

import Helpers.DataHelpers as DataHelpers
from load_data import load_behavior_csv
from plot_results import shaded_curve
import os
import numpy as np
import matplotlib.pyplot as plt

from kernel_regression import (
    kreg_for_aggregate,
    build_hierarchical_data_full,
    hierarchical_bootstrap_joint,
)

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
TRIALS_CSV = "merged_all_subjects.csv"
META_CSV   = "sex_gen.csv"

# -------------------------------
# 1) Load and prepare data
# -------------------------------
df = load_behavior_csv(TRIALS_CSV)
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")

df = df[df["trial_is_repeat"] == False].copy()

df = DataHelpers.restrict_subjects(
    df,
    meta_csv=META_CSV,
    sex=None,
    genotypes=None,
    attach_meta=True,
)

print("Genotypes present:", df["genotype"].unique())

df["absILD"] = df["ILD"].abs()

# -------------------------------
# 2) Kernel regression setup
# -------------------------------
xxi = np.linspace(0.0, 1.0, 200)
h   = 0.015
B   = 1000

# X-axis limits for all plots
XMIN, XMAX = 0.0, 0.5   # <- change these to whatever you like

# -------------------------------
# 3) Group definition
# -------------------------------
group_col    = "genotype"
group_values = ["wt", "het", "hom"]

# -------------------------------
# 4) Run aggregate bootstrap (group-level)
# -------------------------------
RTD_all, TCM_all, CDF_all, MT_all = kreg_for_aggregate(
    df,
    xxi=xxi,
    h=h,
    B=B,
    group_col=group_col,
    group_values=group_values,
    easy_value=None,
    abl_value=None,
)

colors = {
    "wt":  "tab:blue",
    "het": "tab:orange",
    "hom": "tab:green",
}

print("✅ Finished group-level kernel regression. You can now run the plotting cells.")

#%% 
# =====================================================
# 5) Tachometric curves
# =====================================================

fig, ax = plt.subplots(figsize=(7, 4))

for g in group_values:
    med, up, dn = TCM_all[g]
    if med is None or np.isnan(med).all():
        continue
    shaded_curve(xxi, med, up, dn, color=colors[g], label=g)

ax.set_xlabel("RT (s)")
ax.set_ylabel("P(correct)")
ax.set_ylim(0, 1)
ax.set_xlim(XMIN, XMAX)
# ax.grid(axis="x", linestyle=":", alpha=0.5)  # <-- enable if you want vertical grid here too
ax.set_title("Tachometric curves by genotype")
ax.legend()
plt.tight_layout()
plt.show()

#%%
# =====================================================
# 6) MT vs RT (per-group bootstrap)
# =====================================================
# NOTE: This part does additional per-group bootstraps each time you run it,
# because it reuses hierarchical_bootstrap_joint.
# If you want to avoid recomputing, we can also cache MT_g results per group.

fig, ax = plt.subplots(figsize=(7, 4))

for g in group_values:

    data = build_hierarchical_data_full(
        df,
        group_col=group_col,
        group_value=g,
        easy_value=None,
        abl_value=None,
    )

    RTD_g, TCM_g, CDF_g, MT_g = hierarchical_bootstrap_joint(data, xxi, h, B)

    med, up, dn = MT_g
    if med is None or np.isnan(med).all():
        continue

    shaded_curve(xxi, med, up, dn, color=colors[g], label=g)

ax.set_xlabel("RT (s)")
ax.set_ylabel("MT (s)")
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # <-- vertical grid only
ax.set_title("MT vs RT by genotype")
ax.legend()
plt.tight_layout()
plt.show()

#%%
# =====================================================
# 7) RT distribution curves
# =====================================================

fig, ax = plt.subplots(figsize=(7, 4))

for g in group_values:
    med, up, dn = RTD_all[g]
    if med is None or np.isnan(med).all():
        continue
    shaded_curve(xxi, med, up, dn, color=colors[g], label=g)

ax.set_xlabel("RT (s)")
ax.set_ylabel("RT density (arb. units)")
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # <-- vertical grid only
ax.set_title("RTD by genotype")
ax.legend()
plt.tight_layout()
plt.show()

#%%
# =====================================================
# 8) CDF curves
# =====================================================

fig, ax = plt.subplots(figsize=(7, 4))

for g in group_values:
    med, up, dn = CDF_all[g]
    if med is None or np.isnan(med).all():
        continue
    shaded_curve(xxi, med, up, dn, color=colors[g], label=g)

ax.set_xlabel("RT (s)")
ax.set_ylabel("Cumulative probability")
ax.set_ylim(0, 1)
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="x", linestyle=":", alpha=0.5)  # <-- vertical grid only
ax.set_title("RT CDF by genotype")
ax.legend()
plt.tight_layout()
plt.show()
