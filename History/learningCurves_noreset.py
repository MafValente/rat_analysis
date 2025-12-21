#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import Helpers.DataHelpers as DataHelpers

# --- CONFIGURATION ---
SUBJECT_ID_COLUMN = 'animal'         # CHANGE THIS if your subject column is named differently
TRIAL_COLUMN = 'trial'               # CHANGE THIS if your trial column is named differently
SUCCESS_COLUMN = 'success'           # Must be 1 (Correct), -1 (Error), 0 (Incomplete)
SPAN = 20                            # EWMA smoothing span (Try 10, 15, or 20 for different smoothness)
# ---------------------

# Load the data, selecting only required columns to save memory
# The low_memory=False is necessary for large files with mixed data types

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2")
cohort_file = "merged_ASD0008.csv"
df = pd.read_csv(cohort_file, low_memory=False)

df = df[df['training_level'] <= 3]

print(f"✅ Loaded file: {cohort_file} ({len(df)} rows)")

# 1. Calculate the two performance metrics (as binary Series)
# is_completed: 1 if success=1 or -1 (attempted/completed), 0 otherwise (incomplete)
is_completed = ((df[SUCCESS_COLUMN] == 1) | (df[SUCCESS_COLUMN] == -1)).astype(int)

# is_correct: 1 if success=1, 0 otherwise (error or incomplete)
is_correct = (df[SUCCESS_COLUMN] == 1).astype(int)

print(f"✅ 2 performance metrics")

# 2. Create a temporary DataFrame for calculation and plotting
df_plot = pd.DataFrame({
    SUBJECT_ID_COLUMN: df[SUBJECT_ID_COLUMN],
    TRIAL_COLUMN: df[TRIAL_COLUMN],
    'is_completed': is_completed,
    'is_correct': is_correct
})

# ---- SORT BY SUBJECT + TRIAL ----
df_plot = df_plot.sort_values([SUBJECT_ID_COLUMN, TRIAL_COLUMN]).reset_index(drop=True)

print(f"✅ Temporary dataframe")

# 3. Compute Exponentially Weighted Moving Average (EWMA) for smoothing

# ---- COMPUTE EWMA SAFELY ----
def ewm_smooth(x):
    return x.ewm(span=SPAN, adjust=False).mean()

df_plot['smooth_completion'] = (
    df_plot.groupby(SUBJECT_ID_COLUMN)['is_completed']
    .transform(ewm_smooth)
)

df_plot['smooth_accuracy'] = (
    df_plot.groupby(SUBJECT_ID_COLUMN)['is_correct']
    .transform(ewm_smooth)
)


print(f"✅ EWMA")


# 4. Plotting for ALL Subjects

# Get a list of unique subjects
subjects = df_plot[SUBJECT_ID_COLUMN].unique()
num_subjects = len(subjects)

# Setup the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Define colors and marker styles for better differentiation
colors = plt.cm.get_cmap('hsv', num_subjects)

for i, subject in enumerate(subjects):
    df_sub = df_plot[df_plot[SUBJECT_ID_COLUMN] == subject]
    
    # Plotting Completion Curve (Solid Line)
    ax.plot(df_sub[TRIAL_COLUMN], df_sub['smooth_completion'], 
            color=colors(i),
            label=f'{subject} - Completion', 
            linestyle='-', alpha=0.5, linewidth=2)

    # Plotting Accuracy Curve (Dashed Line)
    ax.plot(df_sub[TRIAL_COLUMN], df_sub['smooth_accuracy'], 
            color=colors(i),
            label=f'{subject} - Accuracy', 
            linestyle='--', alpha=0.8, linewidth=2)

# Add performance thresholds
ax.axhline(0.70, color='black', linestyle='-.', alpha=0.5, label='70% Threshold')
ax.axhline(0.90, color='black', linestyle=':', alpha=0.5, label='90% Threshold')

# Set labels and title
ax.set_xlabel('Trial Number', fontsize=14)
ax.set_ylabel(f'Smoothed Rate (EWMA Span={SPAN})', fontsize=14)
ax.set_title(f"Dual Learning Curves for All Subjects (N={num_subjects})", fontsize=16)

# Place legend outside the plot
ax.legend(title='Subject and Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_ylim(0, 1.05)

# Adjust layout for legend and save the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.savefig("all_subjects_learning_curves.png")
plt. show()
print("Plot saved: all_subjects_learning_curves.png")


# %%

"""



 d888b8b    88bd88b d8888b ?88   d8P?88,.d88b, .d888b,
d8P' ?88    88P'  `d8P' ?88d88   88 `?88'  ?88 ?8b,
88b  ,88b  d88     88b  d88?8(  d88   88b  d8P   `?8b
`?88P'`88bd88'     `?8888P'`?88P'?8b  888888P'`?888P'
       )88                            88P'
      ,88P                           d88
  `?8888P                            ?8P
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import Helpers.DataHelpers as DataHelpers

# --- CONFIGURATION ---
SUBJECT_COL = 'animal'         # CHANGE THIS if your subject column is named differently
TRIAL_COLUMN = 'trial'               # CHANGE THIS if your trial column is named differently
SUCCESS_COL = 'success'           # Must be 1 (Correct), -1 (Error), 0 (Incomplete)
SPAN = 25                            # EWMA smoothing span (Try 10, 15, or 20 for different smoothness)
META_CSV = "sex_gen.csv"
TRIAL_COL = "trial"
SESSION_COL = "session"
# ---------------------

# Load the data, selecting only required columns to save memory
# The low_memory=False is necessary for large files with mixed data types

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2")
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file, low_memory=False)

df = df[(df['training_level'] <= 1)]


print(f"✅ Loaded file: {cohort_file} ({len(df)} rows)")
# ===============================================================
#   Compute mean learning curve for a *single* view
# ===============================================================

def compute_view_curve(name, filter_fn, df):
    """
    Computes smoothed learning curves for a subject subset:
      - EWMA smoothed
      - detects session boundaries
      - removes session 1 marker
      - truncates within-group only
      - returns mean + SEM + individual curves + session boundaries
    """

    df_view = filter_fn(df)
    subjects = df_view[SUBJECT_COL].unique()

    if len(subjects) == 0:
        print(f"[WARN] View '{name}' has zero subjects — skipping.")
        return None

    df_view = df_view.copy()

    df_view["is_completed"] = ((df_view[SUCCESS_COL] == 1) | (df_view[SUCCESS_COL] == -1)).astype(int)
    df_view["is_correct"]   = (df_view[SUCCESS_COL] == 1).astype(int)

    # ------------------------------------------------------------
    # Correct sorting: subject → session → trial
    # ------------------------------------------------------------
    df_view = df_view.sort_values([SUBJECT_COL, SESSION_COL, TRIAL_COL]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Detect session starts correctly
    # ------------------------------------------------------------
    df_view["session_start"] = (
        df_view.groupby(SUBJECT_COL)[SESSION_COL]
        .apply(lambda x: x != x.shift())
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------
    # Smooth curves (must NOT reshuffle order again)
    # ------------------------------------------------------------
    def ewm_smooth(x):
        return x.ewm(span=SPAN, adjust=False).mean()

    df_view["smooth_completion"] = df_view.groupby(SUBJECT_COL)["is_completed"].transform(ewm_smooth)
    df_view["smooth_accuracy"]   = df_view.groupby(SUBJECT_COL)["is_correct"].transform(ewm_smooth)

    # ------------------------------------------------------------
    # Collect curves and session markers
    # ------------------------------------------------------------
    comp_curves = []
    acc_curves = []
    session_starts = []

    for s in subjects:
        d = df_view[df_view[SUBJECT_COL] == s]

        comp_curves.append(d["smooth_completion"].values)
        acc_curves.append(d["smooth_accuracy"].values)

        marks = np.where(d["session_start"].values)[0]

        # Remove session 1
        if len(marks) > 0 and marks[0] == 0:
            marks = marks[1:]

        session_starts.append(marks)

    # ------------------------------------------------------------
    # Truncate curves to shortest animal
    # ------------------------------------------------------------
    min_len = min(len(c) for c in comp_curves)

    comp_mat = np.vstack([c[:min_len] for c in comp_curves])
    acc_mat  = np.vstack([c[:min_len] for c in acc_curves])

    truncated_session_starts = [
        marks[marks < min_len] for marks in session_starts
    ]

    # ------------------------------------------------------------
    # Mean and SEM
    # ------------------------------------------------------------
    comp_mean = comp_mat.mean(axis=0)
    comp_sem  = comp_mat.std(axis=0) / np.sqrt(comp_mat.shape[0])

    acc_mean = acc_mat.mean(axis=0)
    acc_sem  = acc_mat.std(axis=0) / np.sqrt(acc_mat.shape[0])

    return {
        "name": name,
        "n": len(subjects),
        "subjects": subjects,
        "length": min_len,
        "completion_mean": comp_mean,
        "completion_sem": comp_sem,
        "accuracy_mean": acc_mean,
        "accuracy_sem": acc_sem,
        "completion_curves": [c[:min_len] for c in comp_curves],
        "accuracy_curves": [c[:min_len] for c in acc_curves],
        "session_starts": truncated_session_starts,
    }



# ===============================================================
#   Define your views (groups)
# ===============================================================

views = [
    ("wt", lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, genotypes="wt",
        subject_col="animal", genotype_col="genotype", attach_meta=True)
     ),

    ("het", lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, genotypes="het",
        subject_col="animal", genotype_col="genotype", attach_meta=True)
     ),

    ("hom", lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, genotypes="hom",
        subject_col="animal", genotype_col="genotype", attach_meta=True)
     ),
]


# ===============================================================
#   Compute all view curves
# ===============================================================

group_results = []
for name, fn in views:
    res = compute_view_curve(name, fn, df)
    if res is not None:
        group_results.append(res)

# ===============================================================
#   Plot all views together
# ===============================================================

plt.figure(figsize=(12, 7))

for g in group_results:
    x = np.arange(g["length"])

    # Mean curve
    plt.plot(x, g["completion_mean"], linewidth=3, label=f"{g['name']} (n={g['n']})")

    # SEM shading
    plt.fill_between(
        x,
        g["completion_mean"] - g["completion_sem"],
        g["completion_mean"] + g["completion_sem"],
        alpha=0.2
    )

plt.title("Learning Curves by View (Completion + SEM)")
plt.xlabel("Trial")
plt.ylabel("Completion Rate")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# for g in group_results:
#     x = np.arange(g["length"])
#     plt.plot(
#         x, g["accuracy"],
#         linewidth=2,
#         label=f"{g['name']} (n={g['n']})"
#     )

# %%
plt.figure(figsize=(12, 7))

for g in group_results:
    x = np.arange(g["length"])

    plt.plot(x, g["accuracy_mean"], linestyle="--", linewidth=3,
             label=f"{g['name']} (n={g['n']})")

    plt.fill_between(
        x,
        g["accuracy_mean"] - g["accuracy_sem"],
        g["accuracy_mean"] + g["accuracy_sem"],
        alpha=0.2
    )

plt.title("Learning Curves by View (Accuracy + SEM)")
plt.xlabel("Trial")
plt.ylabel("Accuracy Rate")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %%

# ===============================================================
#   SECOND FIGURE: Individual curves per group
# ===============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, g in zip(axes, group_results):

    x = np.arange(g["length"])

    # # Individual curves
    # for subj_curve in g["accuracy_curves"]:
    #     ax.plot(x, subj_curve, alpha=0.25, linewidth=1)

    for subj_curve, marks in zip(g["completion_curves"], g["session_starts"]):
        ax.plot(x, subj_curve, alpha=0.25, linewidth=1)

        ax.scatter(marks, subj_curve[marks],
               color="red", s=15, zorder=3)

    # Group mean
    ax.plot(x, g["completion_mean"], color="black", linewidth=2.5)

    # SEM
    ax.fill_between(
        x,
        g["completion_mean"] - g["completion_sem"],
        g["completion_mean"] + g["completion_sem"],
        alpha=0.15
    )

    ax.set_title(f"{g['name']} (n={g['n']})")
    ax.set_xlabel("Trial")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Completion Rate (EWMA)")
plt.suptitle("Individual Completion Curves per Genotype Group")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
