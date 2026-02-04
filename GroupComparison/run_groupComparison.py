#%%
import pickle
import matplotlib as mpl
import os, sys
os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis")
sys.path.insert(0, os.getcwd())

from GroupComparison.config import (
    ViewSpec, FilterConfig, PlotStyle, GroupComparisonConfig,
    OverlaySpec, JNDOverlaySpec
)

from GroupComparison.runner import run_groupcomparison
import Helpers.DataHelpers as DataHelpers

# ----------------- paths -----------------
LINE = "CNTNAP2"
COHORT = "cohort2"
BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"
LINE_ROOTS = {("CNTNAP2", "cohort2"): "CNTNAP2_cohort2", ("SHANK3", "cohort1"): "SHANK3_cohort1"}
DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[(LINE, COHORT)])
os.chdir(DATA_DIR)


# ----------------- font/style -----------------
mpl.rcParams["savefig.pad_inches"] = 0.6
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"]


# ----------------- views -----------------
views = [
    ViewSpec("wt",  lambda d: DataHelpers.restrict_subjects(d, "sex_gen.csv", genotypes="wt",  subject_col="animal", genotype_col="genotype", attach_meta=True)),
    ViewSpec("het", lambda d: DataHelpers.restrict_subjects(d, "sex_gen.csv", genotypes="het", subject_col="animal", genotype_col="genotype", attach_meta=True)),
    ViewSpec("hom", lambda d: DataHelpers.restrict_subjects(d, "sex_gen.csv", genotypes="hom", subject_col="animal", genotype_col="genotype", attach_meta=True)),
]
view_colors = {"wt":  "tab:gray", "het": "tab:olive", "hom": "tab:pink"}


# ----------------- overlays (RT + psychometric) -----------------
with open("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_plot_data.pkl", "rb") as f:
    makefig1_data = pickle.load(f)
with open("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/fig1_chrono_plot_data.pkl", "rb") as f:
    makefig1_chrono = pickle.load(f)

# If you need the 35->40 remap, do it once here (same logic you already use)

overlay = OverlaySpec(
    makefig1_data=makefig1_data,
    makefig1_chrono=makefig1_chrono,
    overlay_color="None",
)

# ----------------- old JND overlay for comparison fig -----------------
old_jnd_path = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/Old Data/ILD_task/jnd_analysis_data.pkl"
with open(old_jnd_path, "rb") as f:
    old_jnd_data = pickle.load(f)

jnd_overlay = JNDOverlaySpec(
    old_jnd_data=old_jnd_data,
    abl_color_map={20: "C0", 40: "C1", 60: "C3"},  # your convention
)


# ----------------- configs -----------------
cfg = GroupComparisonConfig(error_mode="individuals", skip_psy_fits=(50,), ild_shift_for_abl50=True)
fcfg = FilterConfig(
    training_min=16,
    session_min=13,
    drop_repeat_trials=True,
    session_type_values=[1],
    stim_dur_values=[6000],
    sessiontype_or_stimdur="or",   # <- your request
)
style = PlotStyle()


# 1) Layout: views 3x3 + separate JND comparison (old vs new individuals)
out1 = run_groupcomparison(
    cohort_csv="merged_all_subjects.csv",
    views=views,
    cfg=cfg, fcfg=fcfg, style=style,
    overlay=overlay,
    jnd_overlay=jnd_overlay,
    layout="views_3x3",
    view_colors=view_colors,
    show=True,
)


#%%

# 2) Layout: ABLs 4x3 + JND inset inside psychometrics (group mean±SEM per genotype)
out2 = run_groupcomparison(
    cohort_csv="merged_all_subjects.csv",
    views=views,
    cfg=cfg, fcfg=fcfg, style=style,
    overlay=overlay,
    jnd_overlay=jnd_overlay,
    layout="abls_4x3",
    view_colors=view_colors,
    show=True,
)

# %%
