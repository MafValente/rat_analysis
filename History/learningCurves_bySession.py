#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import DataHelpers

# --- CONFIGURATION ---
SUBJECT_COL = 'animal'
TRIAL_COL = 'trial'
SUCCESS_COL = 'success'
SESSION_COL = 'session'
SPAN = 25
META_CSV = "sex_gen.csv"
NORMALIZED_POINTS = 100   # resolution for each session
# ---------------------

# ===============================================================
#   LOAD DATA
# ===============================================================

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/ASD_cohort2")
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file, low_memory=False)

df = df[df["session"] <= 3]

print(f"Loaded file: {cohort_file} ({len(df)} rows)")


# ===============================================================
#   NORMALIZED SESSION-WISE CURVE COMPUTATION
# ===============================================================

def compute_session_curves(name, filter_fn, df):
    """
    Computes *normalized* learning curves per session.
    Each subject's curve runs from 0 → 1 in session time (not trials).
    Per session, curves are resampled to the same number of points.
    """

    df_view = filter_fn(df)
    if len(df_view) == 0:
        print(f"[WARN] View {name} has no subjects.")
        return None

    df_view = df_view.copy()

    # Learning metrics
    df_view["is_correct"] = (df_view[SUCCESS_COL] == 1).astype(int)

    subjects = df_view[SUBJECT_COL].unique()
    sessions = sorted(df_view[SESSION_COL].unique())

    def ewm_smooth(x):
        return x.ewm(span=SPAN, adjust=False).mean()

    output = {"name": name, "sessions": {}}

    # ============================================================
    #   For each session: normalize → resample → aggregate
    # ============================================================
    for sess in sessions:
        df_sess = df_view[df_view[SESSION_COL] == sess]

        sess_curves = []
        sess_subjects = df_sess[SUBJECT_COL].unique()

        for s in sess_subjects:
            sub = df_sess[df_sess[SUBJECT_COL] == s].sort_values(TRIAL_COL)
            smooth = ewm_smooth(sub["is_correct"]).values

            if len(smooth) < 2:
                continue

            # Normalize x to 0–1
            x_raw = np.linspace(0, 1, len(smooth))

            # Resample to uniform length (NORMALIZED_POINTS)
            x_target = np.linspace(0, 1, NORMALIZED_POINTS)
            smooth_resampled = np.interp(x_target, x_raw, smooth)

            sess_curves.append(smooth_resampled)

        if len(sess_curves) == 0:
            continue

        mat = np.vstack(sess_curves)

        mean_curve = mat.mean(axis=0)
        sem_curve = mat.std(axis=0) / np.sqrt(mat.shape[0])

        output["sessions"][sess] = {
            "subjects": sess_subjects,
            "n": len(sess_subjects),
            "length": NORMALIZED_POINTS,
            "mean": mean_curve,
            "sem": sem_curve,
            "curves": sess_curves,
        }

    return output


# ===============================================================
#   DEFINE GENOTYPE VIEWS
# ===============================================================

views = [
    ("wt",  lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, genotypes="wt",  subject_col="animal",
        genotype_col="genotype", attach_meta=True)),

    ("het", lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, genotypes="het", subject_col="animal",
        genotype_col="genotype", attach_meta=True)),

    ("hom", lambda d: DataHelpers.restrict_subjects(
        d, META_CSV, genotypes="hom", subject_col="animal",
        genotype_col="genotype", attach_meta=True)),
]


# ===============================================================
#   COMPUTE NORMALIZED SESSION CURVES
# ===============================================================

session_results = {}
for name, fn in views:
    session_results[name] = compute_session_curves(name, fn, df)




#%%
# ===============================================================
#   PLOT — 3 PANELS (WT | HET | HOM) WITH NORMALIZED SESSIONS
# ===============================================================

# ===============================================================
#   FIGURE — Normalized Sequential Sessions, NO FAKE LINES
# ===============================================================

def plot_normalized_sequential_panels(session_results):

    groups = ["wt", "het", "hom"]
    
    accent = plt.cm.get_cmap("Accent")

    group_colors = {
        "wt":  "black",   # teal-green
        "het": "black",   # strong orange
        "hom": "black",   # light purple
    }

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharey=True)

    for ax, group in zip(axes, groups):

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.set_title(f"{group.upper()} (no data)")
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())
        n_sessions = len(sessions)

        # Build subject → list of segments
        per_subject = {}

        offset = 0
        boundaries = []

        for sess in sessions:
            info = res["sessions"][sess]
            length = info["length"]

            x_norm = np.linspace(offset, offset + 1, length)
            offset += 1
            boundaries.append(x_norm[0])

            # Add data for each subject in session
            for subj, curve in zip(info["subjects"], info["curves"]):
                if subj not in per_subject:
                    per_subject[subj] = []
                per_subject[subj].append((x_norm, curve))

        # Assign colors
        all_subjects = sorted(per_subject.keys())
        cmap = plt.cm.get_cmap("Dark2", len(all_subjects))

        color_map = {subj: cmap(i) for i, subj in enumerate(all_subjects)}

        # ---------- PLOT INDIVIDUAL SUBJECTS ----------
        for subj in all_subjects:
            for x_seg, y_seg in per_subject[subj]:
                ax.plot(
                    x_seg, y_seg,
                    color=color_map[subj],
                    alpha=0.5,
                    linewidth=1.5
                )

            # Create legend entries for subjects
            legend_handles = []
            for subj in all_subjects:
                color = color_map[subj]
                line = plt.Line2D([0], [0], color=color, lw=2, label=subj)
                legend_handles.append(line)

            # Put legend outside right side
            ax.legend(
                handles=legend_handles,
                title="Subjects",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.
            )


        # ---------- GROUP MEAN (resample each segment separately) ----------
        # First, resample each session to equal length
        per_session_means = []
        per_session_sems = []
        per_session_x = []

        for i, sess in enumerate(sessions):
            info = res["sessions"][sess]
            n = info["length"]
            x = np.linspace(i, i + 1, n)

            per_session_x.append(x)
            per_session_means.append(info["mean"])
            per_session_sems.append(info["sem"])

        # Now concatenate means/SEMs in order
        X = np.concatenate(per_session_x)
        MEAN = np.concatenate(per_session_means)
        SEM = np.concatenate(per_session_sems)

        # Plot mean
        ax.plot(X, MEAN, color=group_colors[group], linewidth=3)
        ax.fill_between(X, MEAN-SEM, MEAN+SEM,
                        color=group_colors[group], alpha=0.2)

        # Session boundaries
        for b in boundaries:
            ax.axvline(b, linestyle="--", color="gray", alpha=0.3)

        ax.set_title(f"{group.upper()} — Normalized Sequential Sessions")
        ax.set_xlabel("Session Progress (0–1 per session)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA)")
    plt.suptitle("Normalized Learning Curves per Subject (Corrected, No Artifacts)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ---- CALL THE NEW PLOT ----
plot_normalized_sequential_panels(session_results)

#%%
