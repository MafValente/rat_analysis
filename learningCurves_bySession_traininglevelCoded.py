
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

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file, low_memory=False)

#df = df[df["session"] <= 3]

# Keep only completed trials (drop aborts/incompletes)
df = df[df["success"].isin([1, -1])].copy()

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


    # ("hom", lambda d: DataHelpers.restrict_subjects(
    #     d, META_CSV, genotypes="hom", subject_col="animal",
    #     genotype_col="genotype", attach_meta=True)),
]


# ===============================================================
#   COMPUTE NORMALIZED SESSION CURVES
# ===============================================================

session_results = {}
for name, fn in views:
    session_results[name] = compute_session_curves(name, fn, df)


#%%
"""
.########.##......##..#######......######...#######..##........#######..########..########.....###....########...######.
....##....##..##..##.##.....##....##....##.##.....##.##.......##.....##.##.....##.##.....##...##.##...##.....##.##....##
....##....##..##..##.##.....##....##.......##.....##.##.......##.....##.##.....##.##.....##..##...##..##.....##.##......
....##....##..##..##.##.....##....##.......##.....##.##.......##.....##.########..########..##.....##.########...######.
....##....##..##..##.##.....##....##.......##.....##.##.......##.....##.##...##...##.....##.#########.##...##.........##
....##....##..##..##.##.....##....##....##.##.....##.##.......##.....##.##....##..##.....##.##.....##.##....##..##....##
....##.....###..###...#######......######...#######..########..#######..##.....##.########..##.....##.##.....##..######.
"""
# ===============================================================
#   FINAL PLOT — Mean Curve Colored by Average Training Level
# ===============================================================

def plot_mean_curve_with_two_colorbars(df, session_results):

    # Automatically extract list of group names
    groups = [name for name, _ in views]    

    # Colormaps
    mean_cmap = plt.cm.get_cmap("Set3")     # continuous for training level
    sem_cmap  = plt.cm.get_cmap("coolwarm")  # diverging for variance

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)

    # Track GLOBAL ranges for colorbars
    global_TL_min = np.inf
    global_TL_max = -np.inf
    global_VAR_min = np.inf
    global_VAR_max = -np.inf

    # --------------------------------------------------------------
    #   First pass — compute global min/max for TL and VAR
    # --------------------------------------------------------------
    precomputed = {}

    for group in groups:

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            continue

        sessions = sorted(res["sessions"].keys())

        X_all, MEAN_all, SEM_all = [], [], []
        TL_mean_segments = []
        TL_var_segments  = []

        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            # training-level data
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            # resample training level per animal
            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) < 2:
                    continue
                x_raw = np.linspace(0, 1, len(levels))
                x_tgt = np.linspace(0, 1, n)
                tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)

            TL_mean_segments.append(tlev_mat.mean(axis=0))
            TL_var_segments.append(tlev_mat.var(axis=0))

        TL_mean = np.concatenate(TL_mean_segments)
        TL_var  = np.concatenate(TL_var_segments)

        precomputed[group] = (TL_mean, TL_var)

        # update global ranges
        global_TL_min = min(global_TL_min, TL_mean.min())
        global_TL_max = max(global_TL_max, TL_mean.max())
        global_VAR_min = min(global_VAR_min, TL_var.min())
        global_VAR_max = max(global_VAR_max, TL_var.max())

    # --------------------------------------------------------------
    #   Second pass — plot everything
    # --------------------------------------------------------------
    for ax, group in zip(axes, groups):

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.text(0.5, 0.5, f"{group.upper()} — No data",
                    ha="center", va="center")
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())

        X_all, MEAN_all, SEM_all = [], [], []
        offset = 0

        TL_mean_segments = []
        TL_var_segments  = []

        # ------------- Build arrays again for plotting -------------
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            # x coord for this session
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

            # training-level data
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            # resample training level
            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) < 2:
                    continue
                x_raw = np.linspace(0, 1, len(levels))
                x_tgt = np.linspace(0, 1, n)
                tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)

            TL_mean_segments.append(tlev_mat.mean(axis=0))
            TL_var_segments.append(tlev_mat.var(axis=0))

        # Concatenate full-trial arrays
        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)
        TL_mean = np.concatenate(TL_mean_segments)
        TL_var  = np.concatenate(TL_var_segments)

        # Normalize using global ranges (important!)
        TL_norm = (TL_mean - global_TL_min) / (global_TL_max - global_TL_min + 1e-6)
        VAR_norm = (TL_var - global_VAR_min) / (global_VAR_max - global_VAR_min + 1e-6)

        # ----------------------------------------------------------
        # 1. Plot SEM shading colored by variance
        # ----------------------------------------------------------
        for i in range(len(X) - 1):
            ax.fill_between(
                X[i:i+2],
                MEAN[i:i+2] - SEM[i:i+2],
                MEAN[i:i+2] + SEM[i:i+2],
                color=sem_cmap(VAR_norm[i]),
                alpha=0.25
            )

        # ----------------------------------------------------------
        # 2. Plot mean curve colored by training level
        # ----------------------------------------------------------
        for i in range(len(X) - 1):
            ax.plot(
                X[i:i+2],
                MEAN[i:i+2],
                color=mean_cmap(TL_norm[i]),
                linewidth=3
            )

        # ----------------------------------------------------------
        # 3. Grey individual traces
        # ----------------------------------------------------------
        per_subject = {}
        offset = 0
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1
            for subj, curve in zip(info["subjects"], info["curves"]):
                per_subject.setdefault(subj, {"x": [], "y": []})
                per_subject[subj]["x"].append(x_seg)
                per_subject[subj]["y"].append(curve)

        for subj, data in per_subject.items():
            ax.plot(
                np.concatenate(data["x"]),
                np.concatenate(data["y"]),
                color="lightgray",
                alpha=0.3
            )

        # Session boundaries
        offset = 0
        for sess in sessions:
            ax.axvline(offset, linestyle="--", color="gray", alpha=0.3)
            offset += 1

        ax.set_title(f"{group.upper()} — Mean Learning Curve")
        ax.set_xlabel("Normalized Session Progress (0–1 per session)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA)")

    # ==============================================================
    # Colorbar 1 — Mean Training Level
    # ==============================================================
    sm_TL = plt.cm.ScalarMappable(
        cmap=mean_cmap,
        norm=plt.Normalize(vmin=global_TL_min, vmax=global_TL_max)
    )
    sm_TL.set_array([])

    cax1 = fig.add_axes([1.05, 0.15, 0.03, 0.7])
    cbar1 = fig.colorbar(sm_TL, cax=cax1)
    cbar1.set_label("Average Training Level", fontsize=12)

    # ==============================================================
    # Colorbar 2 — Training-Level Variance
    # ==============================================================
    sm_VAR = plt.cm.ScalarMappable(
        cmap=sem_cmap,
        norm=plt.Normalize(vmin=global_VAR_min, vmax=global_VAR_max)
    )
    sm_VAR.set_array([])

    cax2 = fig.add_axes([1.12, 0.15, 0.03, 0.7])
    cbar2 = fig.colorbar(sm_VAR, cax=cax2)
    cbar2.set_label("Training-Level Variance", fontsize=12)

    # --------------------------------------------------------------
    plt.suptitle("Mean Curves Colored by Training Level (SEM colored by Variance)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    plt.show()



plot_mean_curve_with_two_colorbars(df, session_results)



#%%

"""
..#######..##....##.########.....######...#######..##........#######..########..########.....###....########.
.##.....##.###...##.##..........##....##.##.....##.##.......##.....##.##.....##.##.....##...##.##...##.....##
.##.....##.####..##.##..........##.......##.....##.##.......##.....##.##.....##.##.....##..##...##..##.....##
.##.....##.##.##.##.######......##.......##.....##.##.......##.....##.########..########..##.....##.########.
.##.....##.##..####.##..........##.......##.....##.##.......##.....##.##...##...##.....##.#########.##...##..
.##.....##.##...###.##..........##....##.##.....##.##.......##.....##.##....##..##.....##.##.....##.##....##.
..#######..##....##.########.....######...#######..########..#######..##.....##.########..##.....##.##.....##
"""

def plot_mean_and_individual_training_colored(df, session_results,
                                              color_sem_by_variance=False):
    """
    Plot:
    - Individual subjects colored by their OWN instantaneous training level
    - Group mean colored by AVERAGE training level
    - Optional SEM colored by training-level variance
    """

    TITLE_FONTSIZE = 24
    LABEL_FONTSIZE = 20
    TICK_FONTSIZE  = 16
    CBAR_FONTSIZE  = 18

    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16


    # Automatically extract list of group names
    groups = [name for name, _ in views]

    indiv_cmap = plt.cm.get_cmap("tab20")    # individual traces
    mean_cmap  = plt.cm.get_cmap("tab20")    # mean curve
    sem_cmap   = plt.cm.get_cmap("coolwarm") # SEM (if option enabled)


    fig, axes = plt.subplots(len(groups), 1, figsize=(35, 10), sharey=True)

    # global ranges for color normalization
    global_TL_min = np.inf
    global_TL_max = -np.inf
    global_VAR_min = np.inf
    global_VAR_max = -np.inf

    # -----------------------------------------
    # PASS 1 — compute global ranges
    # -----------------------------------------
    for group in groups:
        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            continue

        TL_all = []
        VAR_all = []

        for sess, info in res["sessions"].items():
            n = info["length"]

            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            # training-level per animal
            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) >= 2:
                    x_raw = np.linspace(0, 1, len(levels))
                    x_tgt = np.linspace(0, 1, n)
                    tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)
            if len(tlev_mat) == 0:
                continue

            TL_all.append(tlev_mat.mean(axis=0))
            VAR_all.append(tlev_mat.var(axis=0))

        if len(TL_all) == 0:
            continue

        TL = np.concatenate(TL_all)
        VAR = np.concatenate(VAR_all)

        global_TL_min = min(global_TL_min, TL.min())
        global_TL_max = max(global_TL_max, TL.max())
        global_VAR_min = min(global_VAR_min, VAR.min())
        global_VAR_max = max(global_VAR_max, VAR.max())

    # -----------------------------------------
    # PASS 2 — plotting
    # -----------------------------------------
    for ax, group in zip(axes, groups):

        res = session_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())

        X_all, MEAN_all, SEM_all = [], [], []
        TL_mean_all, TL_var_all = [], []
        offset = 0

        # ------------------------------
        # Build concatenated structures
        # ------------------------------
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            # session x-range
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

            # training level resampling (group mean + variance)
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            tlev_mat = []
            for subj in info["subjects"]:
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values
                if len(levels) >= 2:
                    x_raw = np.linspace(0, 1, len(levels))
                    x_tgt = np.linspace(0, 1, n)
                    tlev_mat.append(np.interp(x_tgt, x_raw, levels))

            tlev_mat = np.array(tlev_mat)
            TL_mean_all.append(tlev_mat.mean(axis=0))
            TL_var_all.append(tlev_mat.var(axis=0))

        # ------------------------------
        # Concatenate
        # ------------------------------
        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)
        TL_mean = np.concatenate(TL_mean_all)
        TL_var = np.concatenate(TL_var_all)

        # normalize for colorbars
        TL_norm  = (TL_mean - global_TL_min) / (global_TL_max - global_TL_min + 1e-6)
        VAR_norm = (TL_var  - global_VAR_min) / (global_VAR_max - global_VAR_min + 1e-6)

        # -------------------------------------------------------
        # 1. INDIVIDUAL TRACES (COLORED BY EACH OWN TRAINING LVL)
        # -------------------------------------------------------
        per_subject = {}
        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            # get training-level raw data
            df_sess = df[df["session"] == sess]
            df_group = df_sess[df_sess["animal"].isin(info["subjects"])]

            for subj, curve in zip(info["subjects"], info["curves"]):
                dsub = df_group[df_group["animal"] == subj].sort_values("trial")
                levels = dsub["training_level"].values

                if len(levels) < 2:
                    continue

                # resample TL for this subject
                x_raw = np.linspace(0, 1, len(levels))
                x_tgt = np.linspace(0, 1, n)
                TL_subj = np.interp(x_tgt, x_raw, levels)

                per_subject.setdefault(subj, {"x": [], "y": [], "lev": []})
                per_subject[subj]["x"].append(x_seg)
                per_subject[subj]["y"].append(curve)
                per_subject[subj]["lev"].append(TL_subj)

        # plot each subject as a multi-colored line
        for subj, d in per_subject.items():
            Xs = np.concatenate(d["x"])
            Ys = np.concatenate(d["y"])
            Ls = np.concatenate(d["lev"])

            Lnorm = (Ls - global_TL_min) / (global_TL_max - global_TL_min + 1e-6)

            # draw colored fragments
            for i in range(len(Xs)-1):
                ax.plot(
                    Xs[i:i+2],
                    Ys[i:i+2],
                    color=indiv_cmap(Lnorm[i]),
                    linewidth=.5,
                    alpha=0.15
                )

        # -------------------------------------------------------
        # 2. SEM (optional) — color by variance
        # -------------------------------------------------------
        if color_sem_by_variance:
            for i in range(len(X)-1):
                ax.fill_between(
                    X[i:i+2],
                    MEAN[i:i+2] - SEM[i:i+2],
                    MEAN[i:i+2] + SEM[i:i+2],
                    color=sem_cmap(VAR_norm[i]),
                    alpha=0.30
                )
        else:
            ax.fill_between(X, MEAN-SEM, MEAN+SEM, color="lightgray", alpha=0.15)

        # -------------------------------------------------------
        # 3. Mean curve — colored by AVG training level
        # -------------------------------------------------------
        for i in range(len(X)-1):
            ax.plot(X[i:i+2], MEAN[i:i+2],
                    color=mean_cmap(TL_norm[i]),
                    linewidth=1.5)

        # session boundaries
        off = 0
        for sess in sessions:
            ax.axvline(off, linestyle="--", color="gray", alpha=0.3)
            off += 1

        ax.set_title(f"{group.upper()} — Mean + Individual Training-Level Coloring", fontsize=18)
        ax.set_xlabel("Normalized Session Progress (0–1 per session)", fontsize=18)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA)")
    axes[0].set_ylim(.5, 1)
    axes[1].set_ylim(.5, 1)
    axes[2].set_ylim(.5, 1)

    
    # -------------------------------------------------------
    # Colorbar for training level (OUTSIDE the plot)
    # -------------------------------------------------------

    sm1 = plt.cm.ScalarMappable(
        cmap=mean_cmap,
        norm=plt.Normalize(vmin=global_TL_min, vmax=global_TL_max)
    )
    sm1.set_array([])

    # Add new axis to the right side of the figure
    # [left, bottom, width, height] in figure coordinates
    cbar_ax1 = fig.add_axes([1, 0.25, 0.02, 0.5])  
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label("Training Level", fontsize=18)

    # -------------------------------------------------------
    # Colorbar for variance (if used)
    # -------------------------------------------------------
    if color_sem_by_variance:
        sm2 = plt.cm.ScalarMappable(
            cmap=sem_cmap,
            norm=plt.Normalize(vmin=global_VAR_min, vmax=global_VAR_max)
        )
        sm2.set_array([])

        cb2 = fig.colorbar(sm2, ax=axes, location="left", shrink=0.7)
        cb2.set_label("Training-Level Variance")

    plt.tight_layout()
    plt.show()

plot_mean_and_individual_training_colored(df, session_results,
                                          color_sem_by_variance=False)

# %%

"""
.##.......########.##.....##.########.##.............##....#######.
.##.......##.......##.....##.##.......##...........####...##.....##
.##.......##.......##.....##.##.......##.............##...##.......
.##.......######...##.....##.######...##.............##...########.
.##.......##........##...##..##.......##.............##...##.....##
.##.......##.........##.##...##.......##.............##...##.....##
.########.########....###....########.########.....######..#######.


"""


# ===============================================================
#   LOAD DATA
# ===============================================================

os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
cohort_file = "merged_all_subjects.csv"
df = pd.read_csv(cohort_file, low_memory=False)


df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"] == False].copy()

# Keep only completed trials (drop aborts/incompletes)
df = df[df["success"].isin([1, -1])].copy()


print(f"Loaded file: {cohort_file} ({len(df)} rows)")


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


    # ("hom", lambda d: DataHelpers.restrict_subjects(
    #     d, META_CSV, genotypes="hom", subject_col="animal",
    #     genotype_col="genotype", attach_meta=True)),
]


def compute_level16_session_curves(name, filter_fn, df,
                                   span=400, normalized_points=400):
    """
    Computes normalized learning curves for ALL sessions at training level 16,
    with EWMA computed continuously across sessions for each animal
    (no reset at session boundaries).

    Returns:
        {
            "name": group_name,
            "sessions": {
                session_number: {
                    "subjects": [...],
                    "n": int,
                    "length": normalized_points,
                    "mean": np.array,
                    "sem": np.array,
                    "curves": [...],
                }
            }
        }
    """

    # Filter to group
    df_view = filter_fn(df)
    if df_view.empty:
        print(f"[WARN] Group '{name}' has no subjects after filtering.")
        return None

    # Keep ONLY training level 16
    df_lvl = df_view[df_view["training_level"] == 16].copy()
    if df_lvl.empty:
        print(f"[WARN] Group '{name}' has no level-16 data.")
        return None
    
    # Keep only completed trials (drop aborts/incompletes)
    df_lvl = df_lvl[df_lvl["success"].isin([1, -1])].copy()
    if df_lvl.empty:
        print(f"[WARN] Group '{name}' has no completed trials at level 16.")
        return None

    # Compute accuracy
    df_lvl["is_correct"] = (df_lvl["success"] == 1).astype(int)

    # ------------------------------------------------------------
    # EWMA across *all* level-16 trials per animal (no session reset)
    # ------------------------------------------------------------
    def ewm_smooth(x):
        return x.ewm(span=span, adjust=False).mean()

    # sort by animal → session → trial for a proper temporal order
    df_lvl = df_lvl.sort_values(["animal", "session", "trial"])
    df_lvl["smooth_is_correct"] = (
        df_lvl.groupby("animal")["is_correct"]
              .transform(ewm_smooth)
    )

    # Sessions available in level 16
    sessions = sorted(df_lvl["session"].unique())

    output = {"name": name, "sessions": {}}

    # ------------------------------------------------------------
    # Process each level-16 session independently (but using
    # the *continuous* smooth_is_correct)
    # ------------------------------------------------------------
    for sess in sessions:
        df_sess = df_lvl[df_lvl["session"] == sess]
        subjects = sorted(df_sess["animal"].unique())
        curves = []

        for subj in subjects:
            dsub = df_sess[df_sess["animal"] == subj].sort_values("trial")
            smooth = dsub["smooth_is_correct"].values

            if len(smooth) < 2:
                continue

            # Normalize → resample to equal number of points
            x_raw = np.linspace(0, 1, len(smooth))
            x_tgt = np.linspace(0, 1, normalized_points)
            smooth_resampled = np.interp(x_tgt, x_raw, smooth)

            curves.append(smooth_resampled)

        if len(curves) == 0:
            continue

        mat = np.vstack(curves)
        mean_curve = mat.mean(axis=0)
        sem_curve  = mat.std(axis=0) / np.sqrt(mat.shape[0])

        output["sessions"][sess] = {
            "subjects": subjects,
            "n": len(subjects),
            "length": normalized_points,
            "mean": mean_curve,
            "sem": sem_curve,
            "curves": curves,
        }

    return output

level16_results = {
    name: compute_level16_session_curves(name, fn, df)
    for name, fn in views
}

def plot_level16_learning(level16_results):

    # Extract group names automatically
    groups = [name for name in level16_results.keys()]

    fig, axes = plt.subplots(len(groups) + 1, 1,
                             figsize=(9, 3*(len(groups)+1)),
                             sharey=True)

    if len(groups) == 1:
        axes = list(axes)

    group_colors = {group: f"C{i}" for i, group in enumerate(groups)}

    # ----------------- PANELS 1–3 -----------------
    for ax, group in zip(axes[:-1], groups):

        res = level16_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.text(0.5, 0.5, f"{group.upper()}: no level 16 data",
                    ha="center", va="center")
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())

        X_all = []
        MEAN_all = []
        SEM_all = []
        N_per_sess = []

        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            subj_count = info["n"]
            N_per_sess.append((sess, subj_count))

            x_seg = np.linspace(offset, offset + 1, n)
            offset += 1

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

            # Individual thin curves
            for curve in info["curves"]:
                ax.plot(x_seg, curve, color="gray", alpha=0.25, linewidth=1)

        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)

        ax.plot(X, MEAN,
                color=group_colors[group],
                linewidth=1.5,
                label=f"{group.upper()} Mean")

        ax.fill_between(X, MEAN-SEM, MEAN+SEM,
                        color=group_colors[group], alpha=0.15)

        # session markers
        offset = 0
        for sess in sessions:
            ax.axvline(offset, linestyle="--", color="black", alpha=0.2)
            offset += 1

        # annotate n only when it changes
        y_top = ax.get_ylim()[1] * 0.97
        prev_n = None
        for sess, n_animals in N_per_sess:
            if prev_n is None or n_animals != prev_n:
                sess_index = sessions.index(sess)
                x_mid = sess_index + 0.5
                ax.text(x_mid, y_top,
                        f"n={n_animals}",
                        ha="center", va="top",
                        fontsize=10, color="black")
            prev_n = n_animals

        ax.set_title(f"{group.upper()} — Level 16 Across Sessions")
        ax.set_xlabel("Normalized Session Progress (0–1 per session)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA, continuous across sessions)")

    # ----------------- PANEL 4: ALL MEANS -----------------
    ax = axes[-1]

    for group in groups:
        res = level16_results[group]
        if res is None or len(res["sessions"]) == 0:
            continue

        sessions = sorted(res["sessions"].keys())
        X_all, MEAN_all, SEM_all = [], [], []
        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            x_seg = np.linspace(offset, offset+1, n)
            offset += 1

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)

        ax.plot(X, MEAN,
                color=group_colors[group],
                linewidth=1,
                label=group.upper())

        ax.fill_between(X, MEAN-SEM, MEAN+SEM,
                        color=group_colors[group], alpha=0.12)

    ax.set_title("COMBINED — Level 16 Mean Curves for All Groups")
    ax.set_xlabel("Normalized Session Progress (0–1 per session)")
    ax.grid(alpha=0.3)
    ax.set_ylim(.6, 1)
    ax.legend()

    plt.tight_layout()
    plt.show()
# 2) Make the figure
plot_level16_learning(level16_results)

# %%
#non normalized session length

import numpy as np
import matplotlib.pyplot as plt

def compute_level16_session_curves(name, filter_fn, df,
                                   span=200):
    """
    Computes learning curves for ALL sessions at training level 16,
    with EWMA computed continuously across sessions for each animal
    (no reset at session boundaries, no per-session normalization).

    Incomplete/abort trials (e.g. success == 0) are *excluded*:
      - Only success in {1, -1} is used.
      - Accuracy = P(correct | completed).

    Returns:
        {
            "name": group_name,
            "sessions": {
                session_number: {
                    "subjects": [...],           # subjects that contributed
                    "n": int,                   # len(subjects)
                    "length": int,              # truncated length in trials
                    "mean": np.array,           # mean accuracy
                    "sem": np.array,            # standard error
                    "curves": [...],            # list of 1D arrays (per subject)
                }
            }
        }
    """

    # Filter to group
    df_view = filter_fn(df)
    if df_view.empty:
        print(f"[WARN] Group '{name}' has no subjects after filtering.")
        return None

    # Keep ONLY training level 16
    df_lvl = df_view[df_view["training_level"] == 16].copy()
    if df_lvl.empty:
        print(f"[WARN] Group '{name}' has no level-16 data.")
        return None

    # Keep only completed trials (drop aborts/incompletes)
    df_lvl = df_lvl[df_lvl["success"].isin([1, -1])].copy()
    if df_lvl.empty:
        print(f"[WARN] Group '{name}' has no completed trials at level 16.")
        return None

    # Compute accuracy (correct vs error, conditional on completed)
    df_lvl["is_correct"] = (df_lvl["success"] == 1).astype(int)

    # ------------------------------------------------------------
    # EWMA across *all* level-16 completed trials per animal
    # ------------------------------------------------------------
    def ewm_smooth(x):
        return x.ewm(span=span, adjust=False).mean()

    # Ensure temporal order: animal → session → trial
    df_lvl = df_lvl.sort_values(["animal", "session", "trial"])
    df_lvl["smooth_is_correct"] = (
        df_lvl.groupby("animal")["is_correct"].transform(ewm_smooth)
    )

    # Sessions available in level 16
    sessions = sorted(df_lvl["session"].unique())

    output = {"name": name, "sessions": {}}

    # ------------------------------------------------------------
    # Process each session independently, but using continuous EWMA
    # ------------------------------------------------------------
    for sess in sessions:
        df_sess = df_lvl[df_lvl["session"] == sess]

        all_subjects = sorted(df_sess["animal"].unique())
        curves = []
        used_subjects = []

        for subj in all_subjects:
            dsub = df_sess[df_sess["animal"] == subj].sort_values("trial")
            smooth = dsub["smooth_is_correct"].values

            if len(smooth) < 2:
                continue

            curves.append(smooth)
            used_subjects.append(subj)

        if len(curves) == 0:
            continue

        # Truncate to shortest curve in this session (no normalization)
        min_len = min(len(c) for c in curves)
        curves_trunc = [c[:min_len] for c in curves]
        mat = np.vstack(curves_trunc)

        mean_curve = mat.mean(axis=0)
        sem_curve  = mat.std(axis=0) / np.sqrt(mat.shape[0])

        output["sessions"][sess] = {
            "subjects": used_subjects,
            "n": len(used_subjects),
            "length": min_len,
            "mean": mean_curve,
            "sem": sem_curve,
            "curves": curves_trunc,
        }

    return output


# Compute results for each view (e.g. wt / het / hom)
level16_results = {
    name: compute_level16_session_curves(name, fn, df)
    for name, fn in views
}


def plot_level16_learning(level16_results):

    # Extract group names automatically
    groups = [name for name in level16_results.keys()]

    fig, axes = plt.subplots(len(groups) + 1, 1,
                             figsize=(9, 3*(len(groups)+1)),
                             sharey=True)

    if len(groups) == 1:
        axes = list(axes)

    # C0, C1, C2... for group means
    group_colors = {group: f"C{i}" for i, group in enumerate(groups)}

    # ----------------- PANELS 1–(N) -----------------
    for ax, group in zip(axes[:-1], groups):

        res = level16_results[group]
        if res is None or len(res["sessions"]) == 0:
            ax.text(0.5, 0.5, f"{group.upper()}: no level 16 data",
                    ha="center", va="center")
            ax.axis("off")
            continue

        sessions = sorted(res["sessions"].keys())

        X_all = []
        MEAN_all = []
        SEM_all = []
        N_per_sess = []

        offset = 0  # cumulative trial index across sessions

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            subj_count = info["n"]
            N_per_sess.append((sess, subj_count))

            # X axis for this session: actual trial index in this concatenated scale
            x_seg = np.arange(offset, offset + n)
            offset += n

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

            # Individual thin curves
            for curve in info["curves"]:
                ax.plot(x_seg, curve, color="gray", alpha=0.25, linewidth=1)

        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)

        # Group mean
        ax.plot(X, MEAN,
                color=group_colors[group],
                linewidth=1.5,
                label=f"{group.upper()} Mean")

        ax.fill_between(X, MEAN-SEM, MEAN+SEM,
                        color=group_colors[group], alpha=0.15)

        # session markers (at cumulative offsets)
        offset = 0
        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]
            ax.axvline(offset, linestyle="--", color="black", alpha=0.2)
            offset += n

        # annotate n only when it changes
        y_top = ax.get_ylim()[1] * 0.97
        prev_n = None
        offset = 0
        for sess, n_animals in N_per_sess:
            info = res["sessions"][sess]
            n = info["length"]
            x_mid = offset + n / 2.0
            offset += n

            if prev_n is None or n_animals != prev_n:
                ax.text(x_mid, y_top,
                        f"n={n_animals}",
                        ha="center", va="top",
                        fontsize=10, color="black")
            prev_n = n_animals

        ax.set_title(f"{group.upper()} — Level 16 Across Sessions")
        ax.set_xlabel("Completed trials at level 16 (concatenated)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (EWMA, continuous; completed trials only)")

    # ----------------- LAST PANEL: ALL MEANS -----------------
    ax = axes[-1]

    for group in groups:
        res = level16_results[group]
        if res is None or len(res["sessions"]) == 0:
            continue

        sessions = sorted(res["sessions"].keys())
        X_all, MEAN_all, SEM_all = [], [], []
        offset = 0

        for sess in sessions:
            info = res["sessions"][sess]
            n = info["length"]

            x_seg = np.arange(offset, offset + n)
            offset += n

            X_all.append(x_seg)
            MEAN_all.append(info["mean"])
            SEM_all.append(info["sem"])

        X = np.concatenate(X_all)
        MEAN = np.concatenate(MEAN_all)
        SEM = np.concatenate(SEM_all)

        ax.plot(X, MEAN,
                color=group_colors[group],
                linewidth=1,
                label=group.upper())

        ax.fill_between(X, MEAN-SEM, MEAN+SEM,
                        color=group_colors[group], alpha=0.12)

    ax.set_title("COMBINED — Level 16 Mean Curves for All Groups")
    ax.set_xlabel("Completed trials at level 16 (concatenated)")
    ax.set_ylim(.6, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


# 2) Make the figure
plot_level16_learning(level16_results)

# %%
#%%

"""
.########.####.##.....##.########....####.##....##....##.......########.##.....##.########.##......
....##.....##..###...###.##...........##..###...##....##.......##.......##.....##.##.......##......
....##.....##..####.####.##...........##..####..##....##.......##.......##.....##.##.......##......
....##.....##..##.###.##.######.......##..##.##.##....##.......######...##.....##.######...##......
....##.....##..##.....##.##...........##..##..####....##.......##........##...##..##.......##......
....##.....##..##.....##.##...........##..##...###....##.......##.........##.##...##.......##......
....##....####.##.....##.########....####.##....##....########.########....###....########.########
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import DataHelpers


# =======================
# CONFIG
# =======================
SUBJECT_COL = "animal"
LEVEL_COL   = "training_level"
SESSION_COL = "session"
TIME_COL    = "tared_trial_start"  # <-- CHANGE THIS to your actual time column
META_CSV    = "sex_gen.csv"


os.chdir("/Users/mafaldavalente/Documents/Mafalda_analysis/DataFiles/CNTNAP2_cohort2")
df = pd.read_csv("merged_all_subjects.csv", low_memory=False)

df = DataHelpers.prepare_data(df, session_col="session", trial_col="trial")
df = df[df["trial_is_repeat"] == False].copy()

df = df[df["training_level"]<16].copy()

# =======================
# 1) Attach genotype
# =======================

df_all = DataHelpers.restrict_subjects(
    df,
    meta_csv=META_CSV,
    sex=None,
    genotypes=None,             # keep all
    subject_col=SUBJECT_COL,
    genotype_col="genotype",
    attach_meta=True,
)

required = [SUBJECT_COL, LEVEL_COL, SESSION_COL, TIME_COL, "genotype"]
missing = [c for c in required if c not in df_all.columns]
if missing:
    raise KeyError(f"Missing columns in df_all: {missing}")

df_all = df_all.dropna(subset=[LEVEL_COL])

# Make sure time is numeric *seconds within session*
df_all[TIME_COL] = pd.to_numeric(df_all[TIME_COL], errors="coerce")

# =======================
# 2) Per-animal × level × session duration (in seconds)
# =======================

df_all = df_all.sort_values([SUBJECT_COL, SESSION_COL, TIME_COL])

group_sess = df_all.groupby([SUBJECT_COL, "genotype", LEVEL_COL, SESSION_COL])

def _session_level_duration(d):
    """
    Duration spent at this (animal, level, session),
    from first trial at this level in this session
    to last trial at this level in this session.
    """
    t0 = d[TIME_COL].iloc[0]  # seconds from session start
    t1 = d[TIME_COL].iloc[-1]
    dur_sec = float(t1 - t0)  # seconds at that level in this session

    return pd.Series({
        "session_duration_sec": dur_sec,
        "n_trials": len(d),
    })

per_animal_level_session = group_sess.apply(_session_level_duration).reset_index()

print("\n=== Per animal × level × session durations (seconds) ===")
print(per_animal_level_session.head())


# =======================
# 3) Collapse to per-animal × level
# =======================

group_animal_level = per_animal_level_session.groupby(
    [SUBJECT_COL, "genotype", LEVEL_COL]
)

per_animal_level_time = group_animal_level.agg(
    total_time_sec=("session_duration_sec", "sum"),
    mean_session_time_sec=("session_duration_sec", "mean"),
    n_sessions=("session_duration_sec", "size"),
    total_trials=("n_trials", "sum"),
).reset_index()

per_animal_level_time["total_time_min"]  = per_animal_level_time["total_time_sec"] / 60.0
per_animal_level_time["total_time_hour"] = per_animal_level_time["total_time_sec"] / 3600.0

print("\n=== Per animal × level total time (seconds / minutes / hours) ===")
print(
    per_animal_level_time
    .sort_values([LEVEL_COL, "genotype", SUBJECT_COL])
    .head(30)
)

import matplotlib.pyplot as plt

LEVEL_COL = "training_level"

fig, ax = plt.subplots(figsize=(10, 6))

genotype_colors = {"wt": "C0", "het": "C1", "hom": "C2"}

order = ["wt", "het", "hom"]   # whatever order you want

for genotype in order:
    sub = per_animal_level_time[per_animal_level_time["genotype"] == genotype]
    if sub.empty:
        continue


    x = sub[LEVEL_COL].astype(float).values
    y = sub["total_time_hour"].values  # or total_time_min

    jitter = (np.random.rand(len(x)) - 0.5) * 0.25
    x_jittered = x + jitter

    ax.scatter(
        x_jittered,
        y,
        label=genotype,
        color=genotype_colors.get(genotype, "0.5"),
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )

ax.set_xlabel("Training level")
ax.set_ylabel("Total time at level (hours)")  # or minutes
ax.set_title("Time in each training level (per animal)")
ax.grid(alpha=0.3)
ax.legend(title="Genotype")
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

LEVEL_COL = "training_level"
TIME_COL  = "total_time_hour"   # or "total_time_min"

# 1) Convert training_level to integer
per_animal_level_time[LEVEL_COL] = per_animal_level_time[LEVEL_COL].astype(int)

# Ensure training levels are sorted numerically
levels = sorted(per_animal_level_time[LEVEL_COL].unique())

# Genotype order and colors (controls legend order too)
genotypes = ["wt", "het", "hom"]
genotype_colors = {"wt": "C0", "het": "C1", "hom": "C2"}

fig, ax = plt.subplots(figsize=(10, 6))

n_levels = len(levels)
n_genos  = len(genotypes)

# Width of each "cluster" around a level tick
cluster_width = 0.6
# Horizontal step between genotypes within a level
step = cluster_width / n_genos

for g_idx, geno in enumerate(genotypes):
    color = genotype_colors[geno]

    data = []      # list of arrays, one per level
    positions = [] # x-positions for those boxes

    for l_idx, lvl in enumerate(levels):
        vals = per_animal_level_time.loc[
            (per_animal_level_time["genotype"] == geno) &
            (per_animal_level_time[LEVEL_COL] == lvl),
            TIME_COL,
        ].dropna()

        if len(vals) == 0:
            data.append([])  # matplotlib can handle empty entries
        else:
            data.append(vals.values)

        # Center genotypes around integer level index
        center = l_idx
        offset = (g_idx - (n_genos - 1) / 2) * step
        positions.append(center + offset)

    # Plot one boxplot series per genotype
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=step * 0.9,
        patch_artist=True,        # so we can color the box face
        showfliers=True,
        boxprops=dict(facecolor=color, alpha=1, edgecolor=color),
        medianprops=dict(color="black"),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        flierprops=dict(
            marker="o",
            markerfacecolor=color,   # <-- outlier color = genotype color
            markeredgecolor="none",
            markersize=5,
            linestyle="none",
            alpha=1,
        ),
    )

# X-ticks at the level centers
ax.set_xticks(range(n_levels))
ax.set_xticklabels(levels)
ax.set_xlabel("Training level")
ax.set_ylabel("Total time at level (hours)")  # or minutes
ax.set_title("Time spent at each training level (boxplots by genotype)")
ax.grid(axis="y", alpha=0.3)

# Custom legend (one entry per genotype, in the order we chose)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=genotype_colors[g], lw=4, label=g)
    for g in genotypes
]
ax.legend(handles=legend_handles, title="Genotype")

plt.tight_layout()
plt.show()

# %%
