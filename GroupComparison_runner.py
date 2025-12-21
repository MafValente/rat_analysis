# analysis/runners/run_groupcomparison.py
# %%
import os

import Helpers.DataHelpers as DataHelpers
import matplotlib.pyplot as plt

# --- your modular pipeline bits ---
from datasets import DatasetSpec, resolve_data_dir
from pipeline import ViewSpec, AnalysisConfig, build_prepared

# --- plotting package we created ---
from plotting.GroupComparison import plot_groupcomparison
from plotting.GroupComparison.overlays import (
    load_makefig1_data,
    load_makefig1_chrono,
    load_pickle,
)

from views import genotype_views

# ==============================================================
# CONFIG: dataset
# ==============================================================
BASE = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"
spec = DatasetSpec(line="CNTNAP2", cohort="cohort2", base_dir=BASE)

# ==============================================================
# CONFIG: analysis
# ==============================================================
cfg = AnalysisConfig(
    error_mode="individuals",   # "sem" or "individuals"
    training_min=16,
    session_min=13,
    skip_psy_fits=frozenset({50}),  # if you added it to AnalysisConfig; otherwise ignore
)

# ==============================================================
# Views (change only this to switch wt/het/hom, sex splits, etc.)
# ==============================================================
data_dir = resolve_data_dir(spec)
meta_path = os.path.join(data_dir, "sex_gen.csv")

views = genotype_views(spec)
bundle = build_prepared(spec, views, cfg)

# ==============================================================
# Build bundle (NO plotting happens here)
# ==============================================================
bundle = build_prepared(spec, views, cfg)

# Optional sanity prints
#
# print("Prepared views:", list(bundle["prepared"].keys()))
#print("ABL rows:", bundle["abl_rows"])
#for vn in bundle["prepared"]:
   # print(vn, "rt_group rows:", len(bundle["prepared"][vn]["rt_group"]))

# ==============================================================
# Load overlays (old neurotypical reference datasets)
# ==============================================================
makefig1_data_path = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_plot_data.pkl"
makefig1_chrono_path = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_chrono_plot_data.pkl"
old_jnd_path = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/jnd_analysis_data.pkl"

makefig1_data = load_makefig1_data(makefig1_data_path)
makefig1_chrono = load_makefig1_chrono(makefig1_chrono_path)

# Old JND pickle is optional
try:
    old_jnd_data = load_pickle(old_jnd_path)
except FileNotFoundError:
    old_jnd_data = None
    print(f"[warn] old_jnd pickle not found: {old_jnd_path}")


#%%
# ==============================================================
# Plot 1: views as rows (your 1x3-per-view layout)
# ==============================================================

main_fig, jnd_figs = plot_groupcomparison(
    bundle,
    layout="views_1x3",
    makefig1_data=makefig1_data,
    makefig1_chrono=makefig1_chrono,
    old_jnd_data=old_jnd_data,
    show_old_overlays=True,
    show_jnd_inset=True,
    show_old_vs_new_jnd_scatter=True,
    SKIP_PSY_FITS={50},
)

#%%

# ==============================================================
# Plot 2: ABLs as rows (your 4x3-per-ABL layout)
#   This layout is most useful when views include multiple genotypes/sexes.
# ==============================================================
main_fig = plot_groupcomparison(
    bundle,
    layout="abls_4x3",
    makefig1_data=makefig1_data,
    makefig1_chrono=makefig1_chrono,
    show_old_overlays=True,
    SKIP_PSY_FITS={50},

    # xlims per metric:
    xlim_rt={ "default": (0, 19),  50: (17, 19) },   # example
    xlim_mt={ "default": (0, 19), 50: (17, 19) }, # example
    xlim_psy={ "default": (-17, 17), 50: (-19, 19) } # example
)


# %%
