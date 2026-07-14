#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis import psychometric as Psychometric 
import Helpers.DataHelpers as DataHelpers
from matplotlib.ticker import MaxNLocator

subject_file = "merged_ASD0053.csv"

LINE = "CNTNAP2"   # or "SHANK3"
COHORT = "cohort3" # or "cohort1", etc

BASE_DATA_DIR = "/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles"

LINE_ROOTS = {
     ("CNTNAP2", "cohort2"): "CNTNAP2_cohort2",
     ("CNTNAP2", "cohort3"): "CNTNAP2_cohort3",
     ("SHANK3", "cohort1"): "SHANK3_cohort1",
 }

"""



 d888b8b   d8888b  88bd88b d8888b  .d888b, .d888b,
d8P' ?88  d8P' `P  88P'  `d8P' ?88 ?8b,    ?8b,
88b  ,88b 88b     d88     88b  d88   `?8b    `?8b
`?88P'`88b`?888P'd88'     `?8888P'`?888P' `?888P'

                                 d8,
                                `8P

 .d888b, d8888b .d888b, .d888b,  88b d8888b   88bd88b
 ?8b,   d8b_,dP ?8b,    ?8b,     88Pd8P' ?88  88P' ?8b
   `?8b 88b       `?8b    `?8b  d88 88b  d88 d88   88P
`?888P' `?888P'`?888P' `?888P' d88' `?8888P'd88'   88b



"""

# ==============================================================
# CONFIG: choose which line you're analyzing
# ==============================================================


DATA_DIR = os.path.join(BASE_DATA_DIR, LINE_ROOTS[LINE,COHORT])

os.chdir(DATA_DIR)

df = pd.read_csv(subject_file)

colors = ["C0", "C1", "C2", "C3"]

subject_id = subject_file.removeprefix("merged_").removesuffix(".csv")

# 1) Inspect what you actually have
print(df["repeated_trial"].astype(str).str.strip().str.upper().value_counts(dropna=False))

# --- Prep session(s) data
df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")

df = df[df["training_level"]<16].copy()

df_valid = df[df["trial_is_repeat"]==False].copy()

meanRT_per_session = (
    df_valid[(df_valid["success"] == 1) & (df_valid["timed_rt"] <= 1.2)] #ignore outliers
    #.assign(timed_rt=lambda d: d["timed_rt"].clip(upper=1))  # cap RTs at 1s by chamging values
    .groupby(["session", "ABL"])["timed_rt"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

meanMT_per_session = (
    df_valid[(df_valid["success"] == 1) & (df_valid["timed_mt"] <= .8)] 
    #.assign(timed_mt=lambda d: d["timed_mt"].clip(upper=1))  # cap MTs at 1s
    .groupby(["session", "ABL"])["timed_mt"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

meanAcc_per_session = (
    df_valid[(df_valid["success"] != 0)]               # completed trials only
    .groupby(["session", "ABL"])["success"]
    .agg(accuracy=lambda x: (x == 1).mean(),       # proportion correct
         n_trials="count")                         # number of completed trials
    .reset_index()
)


meanFA_per_session = (
    df[df["abort_type"] != "CNP"]               # initiated trials only
    .groupby(["session", "ABL"])["abort_type"]
    .agg(FArate=lambda x: (x == "Fixation").mean(),       # Fraction of Fixation aborts
         n_trials="count")                         # number of completed trials
    .reset_index()
)

# -- Bias independent of correct/incorrect --- currently not plotted
bias_per_session = (
    df_valid.groupby(["session", "ABL"])["response_poke"].mean()
      .reset_index(name="bias")
)

# -- Bias of errors
 
bias_summary = (
    df_valid
      .groupby(["session", "ABL"], observed=False)
      .apply(DataHelpers.compute_bias, include_groups=False)
      .pipe(lambda x: x if isinstance(x, pd.DataFrame) else x.to_frame("bias"))
      .reset_index()
)

meanMTA_per_session = (
    df[df["abort_type"] != "CNP"]               # initiated trials only
    .groupby(["session", "ABL"])["abort_type"]
    .agg(MTArate=lambda x: (x == "MT+").mean(),       # Fraction of long Movement Time aborts
         n_trials="count")                         # number of completed trials
    .reset_index()
)

meanRTA_per_session = (
    df[df["abort_type"] != "CNP"]               # initiated trials only
    .groupby(["session", "ABL"])["abort_type"]
    .agg(RTArate=lambda x: (x == "RT-").mean(),       # Fraction of short Reaction time aborts
         n_trials="count")                         # number of completed trials
    .reset_index()
)

# Repetitions

df_reps = df_valid.sort_values(["session", "trial"])  # make sure trials are ordered
df_reps["prev_response"] = df_reps.groupby("session")["response_poke"].shift(1)
df_reps["repetition"] = (df_reps["response_poke"] == df_reps["prev_response"]).astype(int)


rep_split = (
    df_reps.groupby(["session", "prev_response", "ABL"])["repetition"]
      .mean()
      .reset_index()
      .pivot(index=["session", "ABL"], columns="prev_response", values="repetition")
      .rename(columns={-1: "after_left", 1: "after_right"})
      .reset_index()
)

rep_per_session = (
    df_reps.groupby(["session", "ABL"])["repetition"].mean().reset_index(name="repetition_rate")
)

# trial count per session

completedTrial_count_summary = DataHelpers.count_trials(df_valid, df_valid["success"] != 0, "completed")
CNPA_count_summary = DataHelpers.count_trials(df, df["abort_type"] == "CNP", "cnp")
A_count_summary = DataHelpers.count_trials(df, (df["abort_type"] != "CNP") & (df["success"] == 0), "aborted")

#----- plotting

fig = plt.figure()


plt.plot(
    completedTrial_count_summary["session"],
    completedTrial_count_summary["trial_count"],
    linestyle="-",
    marker="o",
    label="Completed Trials"
    )

plt.plot(
    CNPA_count_summary["session"],
    CNPA_count_summary["trial_count"],
    linestyle="-",
    marker="o",
    label="CNP Aborts"
    )

plt.plot(
    A_count_summary["session"],
    A_count_summary["trial_count"],
    linestyle="-",
    marker="o",
    label="Other Aborts"
    )

# Shade background bands for the current subject

current_subject = subject_id
change_csv = "change_points.csv"       # <- path to your CSV
ax = plt.gca()

if os.path.exists(change_csv):
    regions = DataHelpers.shade_change_regions_from_csv(ax, change_csv, current_subject)
    changes = 1
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")
    changes = 0


ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc="upper center")
plt.xlabel("Session")
plt.ylabel("#trials")
plt.tight_layout()
plt.show()




#%%

fig, axes = plt.subplots(3, 3, figsize=(28, 23))


for i, abl in enumerate(sorted(meanRT_per_session["ABL"].unique())):
    sub = meanRT_per_session[meanRT_per_session["ABL"] == abl]

    axes[0,0].errorbar(
        sub["session"], sub["mean"],
        yerr=sub["std"] / sub["count"]**0.5,  # SEM = std / sqrt(n)
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o", capsize=5
    )


if changes==1:
    DataHelpers.draw_regions(axes[0,0], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")


axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,0].set_xlabel("Session")
axes[0,0].set_ylabel("Reation Time (s)")
#axes[0,0].set_ylim(0, .5)
axes[0,0].set_title("RT progression across sessions")
axes[0,0].legend()

for i, abl in enumerate(sorted(meanMT_per_session["ABL"].unique())):
    sub = meanMT_per_session[meanMT_per_session["ABL"] == abl]

    axes[0,1].errorbar(
        sub["session"], sub["mean"],
        yerr=sub["std"] / sub["count"]**0.5,  # SEM = std / sqrt(n)
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o", capsize=5
    )

if changes==1:
    DataHelpers.draw_regions(axes[0,1], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[0,1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].set_xlabel("Session")
axes[0,1].set_ylabel("Movement Time (s)")
#axes[0,1].set_ylim(0.2, .5)
axes[0,1].set_title("MT progression across sessions")


for i, abl in enumerate(sorted(meanAcc_per_session["ABL"].unique())):
    sub = meanAcc_per_session[meanAcc_per_session["ABL"] == abl]

    axes[0,2].plot(
        sub["session"], sub["accuracy"],
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o"
    )

if changes==1:
    DataHelpers.draw_regions(axes[0,2], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")


axes[0,2].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].set_xlabel("Session")
axes[0,2].set_ylabel("Accuracy (proportion correct)")
axes[0,2].set_ylim(0.5, 1)
axes[0,2].set_title("Accuracy progression across sessions")



for i, abl in enumerate(sorted(meanFA_per_session["ABL"].unique())):
    sub = meanFA_per_session[meanFA_per_session["ABL"] == abl]

    axes[1,0].plot(
        sub["session"], sub["FArate"],
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o"
    )

if changes==1:
    DataHelpers.draw_regions(axes[1,0], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,0].set_xlabel("Session")
axes[1,0].set_ylabel("Proportion of Fixation Aborts")
#axes[1,0].set_ylim(0, .55)
axes[1,0].set_title("Fixation Aborts across sessions")


for i, abl in enumerate(sorted(meanMTA_per_session["ABL"].unique())):
    sub = meanMTA_per_session[meanMTA_per_session["ABL"] == abl]

    axes[1,1].plot(
        sub["session"], sub["MTArate"],
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o"
    )
if changes==1:
    DataHelpers.draw_regions(axes[1,1], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[1,1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,1].set_xlabel("Session")
axes[1,1].set_ylabel("Proportion of MT Aborts")
#axes[1,1].set_ylim(0, .03)
axes[1,1].set_title("Movement Time Aborts across sessions")


for i, abl in enumerate(sorted(meanRTA_per_session["ABL"].unique())):
    sub = meanRTA_per_session[meanRTA_per_session["ABL"] == abl]

    axes[1,2].plot(
        sub["session"], sub["RTArate"],
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o"
    )

if changes==1:
    DataHelpers.draw_regions(axes[1,2], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[1,2].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,2].set_xlabel("Session")
axes[1,2].set_ylabel("Proportion of RT Aborts")
# axes[1,2].set_ylim(0, .025)
axes[1,2].set_title("Reaction Time Aborts across sessions")


#for i, abl in enumerate(sorted(bias_per_session["ABL"].unique())):
#    sub = bias_per_session[bias_per_session["ABL"] == abl]
#
#    axes[2,0].errorbar(
#        sub["session"],
 #       sub["bias"],
  #      color=colors[i % len(colors)],
   #     fmt='o-', capsize=4
    #)

for i, abl in enumerate(sorted(bias_summary["ABL"].unique())):
    sub = bias_summary[bias_summary["ABL"] == abl]

    axes[2,0].errorbar(
        sub["session"],
        sub["bias"],
        color=colors[i % len(colors)],
        fmt='o-', capsize=4
    )

if changes==1:
    DataHelpers.draw_regions(axes[2,0], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[2,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,0].axhline(0, color='k', linestyle='--', alpha=0.7)  # no-bias line
axes[2,0].set_xlabel("Session")
axes[2,0].set_ylabel("Bias (mean response)")
axes[2,0].set_title("Bias across sessions")

for i, abl in enumerate(sorted(rep_per_session["ABL"].unique())):
    sub = rep_per_session[rep_per_session["ABL"] == abl]

    axes[2,1].plot(
        sub["session"], sub["repetition_rate"],
        label=f"ABL {abl}",
        color=colors[i % len(colors)],
        linestyle="-",
        marker="o"
    )

if changes==1:
    DataHelpers.draw_regions(axes[2,1], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[2,1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,1].axhline(0.5, color='k', linestyle='--', alpha=0.7)  # 0.5 = no bias (random)
axes[2,1].set_xlabel("Session")
axes[2,1].set_ylabel("Response repetition rate")
axes[2,1].set_title("Response repetition across sessions")
axes[2,1].set_ylim(0, 1)


for i, abl in enumerate(sorted(rep_split["ABL"].unique())):
    sub = rep_split[rep_split["ABL"] == abl]
    color = colors[i % len(colors)]   # pick color for this ABL

    axes[2,2].plot(sub["session"], sub["after_left"], '<-', color = color)
    axes[2,2].plot(sub["session"], sub["after_right"], '>-', color = color)


if changes==1:
    DataHelpers.draw_regions(axes[2,2], regions, alpha=1)
else:
    # no file → just don't shade anything
    print(f"⚠️ No change_points.csv found at {change_csv}; skipping background shading.")

axes[2,2].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].axhline(0.5, color='k', linestyle='--', alpha=0.7)  # chance level
axes[2,2].set_xlabel("Session")
axes[2,2].set_ylabel("Repetition probability")
axes[2,2].set_title("Response repetition split by previous choice")
axes[2,2].set_ylim(0, 1)

plt.show()



#%%

"""
                          d8b   d8,                                                     d8,
                    d8P   88P  `8P                                                     `8P
                 d888888Pd88
 d8888b ?88   d8P  ?88'  888    88b d8888b  88bd88b     .d888b, d8888b .d888b, .d888b,  88b d8888b   88bd88b  .d888b,
d8P' ?88d88   88   88P   ?88    88Pd8b_,dP  88P'  `     ?8b,   d8b_,dP ?8b,    ?8b,     88Pd8P' ?88  88P' ?8b ?8b,
88b  d88?8(  d88   88b    88b  d88 88b     d88            `?8b 88b       `?8b    `?8b  d88 88b  d88 d88   88P   `?8b
`?8888P'`?88P'?8b  `?8b    88bd88' `?888P'd88'         `?888P' `?888P'`?888P' `?888P' d88' `?8888P'd88'   88b`?888P'



"""

sessions_ok, report, excluded, df_clean = DataHelpers.filter_sessions_with_history_bias_and_perf(
    df_trials=df,             # your trial-level table
    min_trials=100,           # trial count threshold
    trials_use="total",       # or "valid" to use only success!=0 for min_trials
    history_n=10,             # look back over previous sessions per subject for bias cutoff
    min_prev=5,               # need at least this many previous sessions to enforce bias
    k=3.0,                    # MAD multiplier for bias cutoff
    min_perf=0.7,             # e.g., require ≥70% correct among completed
    require_perf_min_completed=20  # only enforce perf if ≥20 completed trials in that session
)

print("Excluded sessions:")
for line in report:
    print("-", line)
