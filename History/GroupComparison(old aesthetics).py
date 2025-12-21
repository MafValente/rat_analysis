#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Psychometric 
import Helpers.DataHelpers as DataHelpers

"""
..######...########...#######..##.....##.########......######...#######..##.....##.########.....###....########..####..######...#######..##....##
.##....##..##.....##.##.....##.##.....##.##.....##....##....##.##.....##.###...###.##.....##...##.##...##.....##..##..##....##.##.....##.###...##
.##........##.....##.##.....##.##.....##.##.....##....##.......##.....##.####.####.##.....##..##...##..##.....##..##..##.......##.....##.####..##
.##...####.########..##.....##.##.....##.########.....##.......##.....##.##.###.##.########..##.....##.########...##...######..##.....##.##.##.##
.##....##..##...##...##.....##.##.....##.##...........##.......##.....##.##.....##.##........#########.##...##....##........##.##.....##.##..####
.##....##..##....##..##.....##.##.....##.##...........##....##.##.....##.##.....##.##........##.....##.##....##...##..##....##.##.....##.##...###
..######...##.....##..#######...#######..##............######...#######..##.....##.##........##.....##.##.....##.####..######...#######..##....##
"""
colors = ["C0", "C1", "C2", "C3"]

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2")
cohort_file = "merged_all_subjects.csv"

"""
        d8b                                      d8b
        88P                                      88P              d8P
       d88                                      d88            d888888P
 d8888b888   d8888b d888b8b    88bd88b      d888888   d888b8b    ?88'   d888b8b
d8P' `P?88  d8b_,dPd8P' ?88    88P' ?8b    d8P' ?88  d8P' ?88    88P   d8P' ?88
88b     88b 88b    88b  ,88b  d88   88P    88b  ,88b 88b  ,88b   88b   88b  ,88b
`?888P'  88b`?888P'`?88P'`88bd88'   88b    `?88P'`88b`?88P'`88b  `?8b  `?88P'`88b



"""

df = pd.read_csv(cohort_file)

#Mask ABL 59 to 60dB
mask2 = df["ABL"] == 59
df.loc[mask2, "ABL"] = 60

#Mask lateralized sound as 50dB
mask3 = (df["training_level"] == 16) & (df["ABL"] == 25)
df.loc[mask3, "ABL"] = 50

#Restrict training level
df = df[(df["training_level"] >= 13)]


#Restrict session number
df = df[df["session"]>=13] # restricting session!!


# 1) define “views” (each returns a restricted copy)
views = [
    ("Female wt", lambda d: DataHelpers.restrict_subjects(
        d, "sex_gen.csv", sex="female", genotypes="wt", subject_col="animal", genotype_col="genotype", attach_meta=True)),
    ("Male wt",   lambda d: DataHelpers.restrict_subjects(
        d, "sex_gen.csv", sex="male", genotypes="wt",  subject_col="animal", genotype_col="genotype", attach_meta=True)),
]



multi_view = len(views) > 1

"""
..######...#######..##.....##.########....########.....###.....######..##....##....##.....##.########.########..########
.##....##.##.....##.###...###.##..........##.....##...##.##...##....##.##...##.....##.....##.##.......##.....##.##......
.##.......##.....##.####.####.##..........##.....##..##...##..##.......##..##......##.....##.##.......##.....##.##......
.##.......##.....##.##.###.##.######......########..##.....##.##.......#####.......#########.######...########..######..
.##.......##.....##.##.....##.##..........##.....##.#########.##.......##..##......##.....##.##.......##...##...##......
.##....##.##.....##.##.....##.##..........##.....##.##.....##.##....##.##...##.....##.....##.##.......##....##..##......
..######...#######..##.....##.########....########..##.....##..######..##....##....##.....##.########.##.....##.########

# ==== Who contributed? ====

# Overall (after your filtering)
contributors_overall = sorted(df[df["success"] == 1]["animal"].unique())
print(f"Overall contributors ({len(contributors_overall)}):", contributors_overall)

# By ABL (list + counts)
contributors_by_abl = (
    df[df["success"] == 1]
    .groupby("ABL")["animal"]
    .agg(lambda s: sorted(s.unique()))
    .to_dict()
)
print("\nContributors by ABL:")
for abl, subs in contributors_by_abl.items():
    print(f"  ABL {abl} (n={len(subs)}): {subs}")

# By box 
contributors_by_setup = (
    df[df["success"] == 1]
    .groupby(["box"])["animal"]
    .agg(subjects=lambda s: sorted(s.unique()),
         n_subjects=lambda s: s.nunique())
    .reset_index()
)
#print("\nContributors by setup:")
#print(contributors_by_setup.head())
"""

# Step 1: compute RT 
def prep_rt(df):
    per_subj = (
        df[df["success"] == 1]
        .groupby(["animal","ABL","ILD"])["timed_rt"]
        .agg(mean_rt="mean")
        .reset_index()
    )
    grouped = (
        per_subj.groupby(["ABL","ILD"])["mean_rt"]
        .agg(mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(len(x)))
        .reset_index()
    )
    return grouped


# Step 1: compute MT 
def prep_mt(df):
    per_subj = (
        df[df["success"] == 1]
        .groupby(["animal","ABL","ILD"])["timed_mt"]
        .agg(mean_mt="mean")
        .reset_index()
    )
    grouped = (
        per_subj.groupby(["ABL","ILD"])["mean_mt"]
        .agg(mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(len(x)))
        .reset_index()
    )
    return grouped



# Step 1: compute PropRight

def prep_psy(df):
    # build points_df (your loop per subject)
    all_pts = []
    for subject, df_subj in df.groupby("animal"):
        results = Psychometric.compute_psychometrics_by_ABL(df_subj, model="erf_psycho")
        for abl, res in results.items():
            for ild, pleft in zip(res["ILDs"], res["PropLeft"]):
                all_pts.append({"subject":subject,"ABL":abl,"ILD":ild,"PropLeft":pleft})
    points = pd.DataFrame(all_pts)

    agg = (points.groupby(["ABL","ILD"])["PropLeft"]
           .agg(mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(len(x)))
           .reset_index())

    # fit per ABL (same as your code, but return a dict)
    fits = {}
    for abl in sorted(agg["ABL"].unique()):
        sub = agg[agg["ABL"] == abl]
        ILDs = sub["ILD"].values
        y = sub["mean"].values
        n_trials = np.full_like(ILDs, 50)
        pars, L, xx, yy = Psychometric.fit_and_plot_psychometric(
            ILDs, y, model="erf_psycho", n_trials=n_trials, show_plot=False
        )
        xx = np.linspace(ILDs.min(), ILDs.max(), 200)
        fits[abl] = dict(ILDs=ILDs, mean=y, sem=sub["sem"].values, xx=xx, yy=yy)
    return agg, fits


prepared = {}  # view_name -> dict of tables
for view_name, make_view in views:
    df_v = df            # your level/ABL remaps etc.
    df_v = make_view(df_v)              # restriction specific to the view
    prepared[view_name] = {
        "rt": prep_rt(df_v),            # columns: ABL, ILD, mean, sem
        "mt": prep_mt(df_v),
        "psy": prep_psy(df_v),          # returns (agg, fits)
    }

#---------- Plotting

# ---- decide ABL rows from prepared data
abl_set = set()
for p in prepared.values():
    abl_set |= set(p["rt"]["ABL"].unique())  # enough to drive rows
abl_rows = sorted(abl_set)

# ---- choose layout mode (single-row if only 1 view OR only 1 ABL)
multi_ABL = len(abl_rows) > 1
single_row_mode = not (multi_view and multi_ABL)

if single_row_mode:
    view_name, _ = views[0]
    rt = prepared[view_name]["rt"]
    mt = prepared[view_name]["mt"]
    psy_agg, psy_fits = prepared[view_name]["psy"]

    abls = sorted(set(rt["ABL"]).union(mt["ABL"]).union(psy_agg["ABL"]))
    abl_colors = {abl: colors[i % len(colors)] for i, abl in enumerate(abls)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RT
    ax = axes[0]
    for abl in abls:
        sub = rt[rt["ABL"] == abl]
        if sub.empty: continue
        ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"], fmt="o", capsize=5,
                    color=abl_colors[abl], label=f"ABL {abl}")
    ax.set(title="RT", xlabel="ILD", ylabel="mean RT (s)")
    ax.legend()

    # MT
    ax = axes[1]
    for abl in abls:
        sub = mt[mt["ABL"] == abl]
        if sub.empty: continue
        ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"], fmt="o", capsize=5,
                    color=abl_colors[abl], label=f"ABL {abl}")
    ax.set(title="MT", xlabel="ILD", ylabel="mean MT (s)")

    # Psychometric
    ax = axes[2]
    for abl in abls:
        sub = psy_agg[psy_agg["ABL"] == abl]
        if sub.empty: continue
        ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"], fmt="o", capsize=3,
                    color=abl_colors[abl], label=f"ABL {abl}")
        fit = psy_fits[abl] if abl in psy_fits else None
        if fit:
            ax.plot(fit["xx"], fit["yy"], color=abl_colors[abl])
    ax.set(title="Psychometric", xlabel="ILD", ylabel="Proportion Left", ylim=(0,1))

    fig.tight_layout()
    plt.show()
    pass
else:
    # ---- multi-view grid: rows = ABLs, cols = metrics
    fig, axes = plt.subplots(len(abl_rows), 3, figsize=(15, 4*len(abl_rows)), squeeze=False)

    #view_colors  = {name: colors[i % len(colors)] for i, (name, _) in enumerate(views)}
    FEMALE = "#e75480"   # pink
    MALE   = "#1f77b4"   # blue (matplotlib default blue)

    preferred = {
        "All females": FEMALE,
        "All males":   MALE,
    }

    view_colors = {
        name: preferred.get(name, colors[i % len(colors)])   # fallback cycles C0,C1,...
        for i, (name, _) in enumerate(views)
    }
    view_markers = ["o", "s", "^", "D", "v"]

    for r, abl in enumerate(abl_rows):
        # RT
        ax = axes[r, 0]
        for vi, (view_name, _) in enumerate(views):
            rt = prepared[view_name]["rt"]
            sub = rt[rt["ABL"] == abl]
            if sub.empty: continue
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt=view_markers[vi % len(view_markers)], capsize=4,
                        color=view_colors[view_name],
                        label=view_name if r == 0 else None, linestyle="none")
        ax.set(title=f"ABL {abl} — RT", xlabel="ILD", ylabel="mean RT (s)")

        # MT
        ax = axes[r, 1]
        for vi, (view_name, _) in enumerate(views):
            mt = prepared[view_name]["mt"]
            sub = mt[mt["ABL"] == abl]
            if sub.empty: continue
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt=view_markers[vi % len(view_markers)], capsize=4,
                        color=view_colors[view_name], linestyle="none")
        ax.set(title=f"ABL {abl} — MT", xlabel="ILD", ylabel="mean MT (s)")

        # Psychometric
        ax = axes[r, 2]
        skip_fit = {50}
        for vi, (view_name, _) in enumerate(views):
            psy_agg, psy_fits = prepared[view_name]["psy"]
            if abl not in psy_agg["ABL"].unique(): continue
            sub = psy_agg[psy_agg["ABL"] == abl]
            ax.errorbar(sub["ILD"], sub["mean"], yerr=sub["sem"],
                        fmt=view_markers[vi % len(view_markers)], capsize=3,
                        color=view_colors[view_name])
            fit = psy_fits.get(abl)
            if fit and abl not in skip_fit:
                ax.plot(fit["xx"], fit["yy"], color=view_colors[view_name])
        ax.set(title=f"ABL {abl} — Psychometric", xlabel="ILD", ylabel="Proportion Left", ylim=(0, 1))



    # one legend for all views
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(views)))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
# %%
