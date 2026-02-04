
#%%
# run_stimdurComparison.py
# ==============================================================
# StimDur comparison (session_type == 2)
# Layout: 4x3 (rows=ABL; cols=RT/MT/Psychometric)
# Lines: stim_dur
# Views: genotype (wt/het/hom) using DataHelpers.restrict_subjects
# Each view produces its own figure.
# ==============================================================

import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

COHORT_CSV = "merged_ASD0021.csv"   # resolved relative to DATA_DIR

# --------------------------------------------------------------
# Project root on path (same style as GroupComparison)
# --------------------------------------------------------------
PROJECT_ROOT = "/Users/mafaldavalente/Documents/Mafalda_analysis"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.getcwd())

import Helpers.DataHelpers as DataHelpers

# --------------------------------------------------------------
# Import StimDur package
# --------------------------------------------------------------
from StimDur.config import (
    ViewSpec, FilterConfig, PlotStyle, StimDurComparisonConfig,
    make_stimdur_specs
)
from StimDur.runner import run_stimdur_comparison
from StimDur.layouts import (plot_genotypes_4x3_for_stimdur,
                            plot_absild_perf_across_stimdur_1x3_for_view, 
                            plot_kreg_4x3_by_abl_for_view
)

from StimDur.prepare import apply_filters

# ----------------- paths -----------------
LINE = "CNTNAP2"
COHORT = "cohort2"

BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"
LINE_ROOTS = {
    ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
    ("SHANK3", "cohort1"): "SHANK3_cohort1",
}
DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[(LINE, COHORT)])
os.chdir(DATA_DIR)




# ----------------- font/style -----------------
mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]


# ----------------- views (genotype/animal selectors) -----------------

animals = (
    pd.read_csv(COHORT_CSV, usecols=["animal"])["animal"]
    .dropna()
    .astype(str)
    .unique()
)

if len(animals) == 1:
    ANIMAL = animals[0]
    views = [
        ViewSpec(
            str(ANIMAL),
            lambda d, a=ANIMAL: d[d["animal"].astype(str) == a].copy()
        )
    ]
else:
    views = [
        ViewSpec("wt",  lambda d: DataHelpers.restrict_subjects(
            d, "sex_gen.csv",
            genotypes="wt",
            subject_col="animal", genotype_col="genotype",
            attach_meta=True
        )),
        ViewSpec("het", lambda d: DataHelpers.restrict_subjects(
            d, "sex_gen.csv",
            genotypes="het",
            subject_col="animal", genotype_col="genotype",
            attach_meta=True
        )),
        ViewSpec("hom", lambda d: DataHelpers.restrict_subjects(
            d, "sex_gen.csv",
            genotypes="hom",
            subject_col="animal", genotype_col="genotype",
            attach_meta=True
        )),
    ]


# ----------------- stim_dur traces -----------------
# IMPORTANT: update these to match what's in your session_type==2 data
STIMDUR_COL = "stim_dur"
STIM_DURS = [15, 60, 120, 6000]  
stimdur_pretty = {
    "15": "SD = 15 ms",
    "60": "SD = 60 ms",
    "120": "SD = 120 ms",
    "6000": "SD = RT",
}

stimdur_specs = make_stimdur_specs(STIM_DURS, stim_dur_col=STIMDUR_COL)

# Colors for stimdur lines (different from view_colors in GroupComparison)
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]  # blue, green, vermillion, purple

stimdur_colors = {s.name: PALETTE[i % len(PALETTE)] for i, s in enumerate(stimdur_specs)}



# ----------------- configs -----------------
# This is the key difference from GroupComparison:
# session_type_values must be [2] and stim_dur is NOT filtered here.
cfg = StimDurComparisonConfig(
    error_mode="individuals",
    skip_psy_fits=(50,),
    ild_shift_for_abl50=True
)
fcfg = FilterConfig(
    training_min=16,
    session_min=13,
    drop_repeat_trials=True,
    session_type_values=[2],   # <-- session_type restriction
)
style = PlotStyle()


# ==============================================================
# RUN (one 4x3 figure per view)
# ==============================================================
out = run_stimdur_comparison(
    cohort_csv=COHORT_CSV,
    views=views,
    stimdur_specs=stimdur_specs,
    cfg=cfg,
    fcfg=fcfg,
    style=style,
    stimdur_colors=stimdur_colors,
    stimdur_pretty=stimdur_pretty,
    show=True,
)

# ==============================================================
# Optional: save figures (one per view)
# ==============================================================
# SAVE_DIR = os.path.join(PROJECT_ROOT, "Figures", "StimDurComparison", f"{LINE}_{COHORT}")
# os.makedirs(SAVE_DIR, exist_ok=True)

# for view_name, fig in out["figures"].items():
#     outpath = os.path.join(SAVE_DIR, f"stimdur_4x3_{view_name}.pdf")
#     fig.savefig(outpath, bbox_inches="tight")
#     print("[Saved]", outpath)

plt.show()

# %%
# ==============================================================
# RUN (one 4x3 figure per SD)
# ==============================================================
view_colors = {
    "wt":  "C0",
    "het": "C1",
    "hom": "C2",
}

figs_by_stimdur = {}
for s in stimdur_specs:
    figs_by_stimdur[s.name] = plot_genotypes_4x3_for_stimdur(
        prepared=out["prepared"],
        group_jnd=out["group_jnd"],
        views=views,
        stimdur_name=s.name,
        view_colors=view_colors,
        cfg=cfg,
        style=style,
        stimdur_pretty=stimdur_pretty,
        view_pretty={"wt": "WT", "het": "HET", "hom": "HOM"},
    )
    plt.show()


# %%

# ================================================================
# GROUP (one 3X1 figure) performance for each ILD over the StimDurs
# ================================================================


for v in views:
    fig = plot_absild_perf_across_stimdur_1x3_for_view(
        prepared_for_view=out["prepared"][v.name]["df_view"],
        stimdur_specs=stimdur_specs,
        view_name=v.name,
        cfg=cfg,
        style=style,
        stimdur_pretty=stimdur_pretty,
        abls=[20, 40, 60],   # choose your 3 ABLs
    )
    plt.show()

# %%


for v in views:
    fig = plot_kreg_4x3_by_abl_for_view(
        df_view=out["df_by_view"][v.name],
        view_name=v.name,
        stimdur_specs=stimdur_specs,
        stimdur_col=STIMDUR_COL,
        stimdur_colors=stimdur_colors,
        stimdur_pretty=stimdur_pretty,
        abls=(20, 40, 60),
        xlim=(0.0, 0.5),
        debug=True,
    )
    plt.show()

# %%
